import argparse
import torch
import torch.nn as nn

import time

from networks import get_network
from utils.loading import parse_spec
from transforms import propagate_linear_symbolic

DEVICE = "cpu"


def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:

    lb = torch.clamp(inputs - eps, 0, 1)
    ub = torch.clamp(inputs + eps, 0, 1)

    # Idea: for each node in the network, compute its function of the input,
    #       for example for node i in layer l f_il(x) = w_il1 * x_1 + ... + w_iln * x_n + b_il
    #       where x is the input of the network

    # Propagate these functions till the output layer, then we get functions in x_1, ..., x_n for 
    # each output neuron. Then we can compute the bounds for each output neuron and check if the
    # true label is the largest output neuron.

    # THIS IS NOT DEEP_POLY! 

    # identity for input layer
    inputs = torch.eye(lb.numel())
    # TODO: make this work for multiple channels
    inputs = torch.reshape(inputs, (lb.shape[1], lb.shape[2], lb.shape[1] * lb.shape[2]))
    # add constant 0 to each row for bias
    shape = inputs.shape[:-1] + (1,)
    inputs = torch.cat((inputs, torch.zeros(shape)), dim=-1)

    # propagate box through network
    for layer in net:
        if isinstance(layer, nn.Flatten):
            inputs = torch.flatten(inputs, start_dim=0, end_dim=1)
        elif isinstance(layer, nn.ReLU):
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
        elif isinstance(layer, nn.Linear):
            inputs = propagate_linear_symbolic(inputs, layer)
        elif isinstance(layer, nn.Conv2d):
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')

    # check if output condition is violated
    lb = lb.flatten()
    ub = ub.flatten()

    ub_out = torch.zeros(inputs.shape[0])
    lb_out = torch.zeros(inputs.shape[0])
    # given input matrix and input bounds compute output bounds
    for i in range(inputs.shape[0]):
        row = inputs[i,:-1]
        b = inputs[i,-1]
        ub_temp = ub.detach().clone()
        ub_temp[row < 0] = lb[row < 0]
        lb_temp = lb.detach().clone()
        lb_temp[row < 0] = ub[row < 0]
        ub_out[i] = row @ ub_temp + b
        lb_out[i] = row @ lb_temp + b

    # check post-condition
    ub_out[true_label] = -float("inf")
    return lb_out[true_label] > ub_out.max()


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
