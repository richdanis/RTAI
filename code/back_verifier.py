import argparse
import torch
import torch.nn as nn

import time

from networks import get_network
from utils.loading import parse_spec
from back_transforms import *

DEVICE = "cpu"

def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    
    # input shape:
    # mnist (1, 28, 28) -> x_1 , ..., x_784
    # cifar10 (3, 32, 32) -> 1_1, ... , x_3072

    # SETUP

    #print(net)

    # set weights to no_grad:
    for param in net.parameters():
        param.requires_grad = False

    init_lb = torch.clamp(inputs - eps, 0, 1)
    init_ub = torch.clamp(inputs + eps, 0, 1)

    shape = init_lb.shape

    init_lb = init_lb.view((-1,))
    init_ub = init_ub.view((-1,))

    bounds = []

    # FORWARD PASS

    # propagate box through network
    for layer in net:
        if isinstance(layer, nn.Flatten):
            pass
        elif isinstance(layer, nn.Linear):
            bounds.append(transform_linear(layer.weight, layer.bias))
        elif isinstance(layer, nn.Conv2d):
            conv_bounds, shape = transform_conv2d(layer, shape)
            bounds.append(conv_bounds)
        elif isinstance(layer, nn.ReLU):
            curr_bound = back(bounds.copy())
            lb, ub = eval(curr_bound, init_lb, init_ub)
            alpha = torch.where(ub.abs() > lb.abs(), torch.ones_like(lb), torch.zeros_like(lb))
            bounds.append(transform_ReLU_alpha(lb, ub, alpha))
        elif isinstance(layer, nn.LeakyReLU):
            pass
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')

    # ADD LAST LAYER
    mat = -torch.eye(10)
    mat[:,true_label] = 1
    mat[true_label,true_label] = 0

    bounds.append(transform_linear(mat, torch.zeros(10)))

    # CHECK VERIFICATION
    bound = back(bounds)
    lb, ub = eval(bound, init_lb, init_ub)
    return int(lb.min() >= 0)


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
