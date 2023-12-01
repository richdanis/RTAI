import argparse
import torch
import torch.nn as nn

import time

from networks import get_network
from utils.loading import parse_spec
from symbolic_transforms import propagate_linear_symbolic, propagate_conv2d_symbolic

DEVICE = "cpu"


def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    
    # input shape:
    # mnist (1, 28, 28)
    # cifar10 (3, 32, 32)

    lb = torch.clamp(inputs - eps, 0, 1)
    ub = torch.clamp(inputs + eps, 0, 1)

    # identity for input layer
    inputs = torch.eye(lb.numel())
    inputs = torch.reshape(inputs, (lb.shape[0], lb.shape[1], lb.shape[2], lb.shape[1] * lb.shape[2]))
    # add constant 0 for bias
    shape = inputs.shape[:-1] + (1,)
    inputs = torch.cat((inputs, torch.zeros(shape)), dim=-1)

    # propagate box through network
    for layer in net:
        # if sequential, list modules of the sequential
        if isinstance(layer, nn.Flatten):
            inputs = torch.flatten(inputs, start_dim=0, end_dim=-2)
        elif isinstance(layer, nn.Linear):
            inputs = propagate_linear_symbolic(inputs, layer.weight, layer.bias)
        elif isinstance(layer, nn.Conv2d):
            inputs = propagate_conv2d_symbolic(inputs, layer)
        elif isinstance(layer, nn.ReLU):
            # treat as identity (bad over approximation)
            pass
        elif isinstance(layer, nn.LeakyReLU):
            # treat as identity (bad over approximation)
            pass
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')

    lb = lb.flatten()
    ub = ub.flatten()

    # at the end inputs is always of shape (10, num_symbols + 1)

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
