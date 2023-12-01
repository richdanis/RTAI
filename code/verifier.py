import argparse
import torch
import torch.nn as nn

import time

from networks import get_network
from utils.loading import parse_spec
from symbolic_transforms import check_postcondition
from box_transforms import propagate_linear_box, propagate_conv2d_box

DEVICE = "cpu"


def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:

    box_lbs = [torch.clamp(inputs - eps, 0, 1)]
    box_ubs = [torch.clamp(inputs + eps, 0, 1)]

    verified = False

    # forward pass for box
    for layer in net:
        if isinstance(layer, nn.Flatten):
            box_lbs.append(torch.flatten(box_lbs[-1]))
            box_ubs.append(torch.flatten(box_ubs[-1]))
        elif isinstance(layer, nn.ReLU):
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
        elif isinstance(layer, nn.Linear):
            lb, ub = propagate_linear_box(box_lbs[-1], box_ubs[-1], layer)
            box_lbs.append(lb)
            box_ubs.append(ub)
        elif isinstance(layer, nn.Conv2d):
            lb, ub = propagate_conv2d_box(box_lbs[-1], box_ubs[-1], layer)
            box_lbs.append(lb)
            box_ubs.append(ub)
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
        
    verified = check_postcondition(box_lbs[-1], box_ubs[-1], true_label)
    return int(verified)
    lb, ub = box_lbs.pop(), box_ubs.pop()

    # backsubstitution
    # iterate backwards through layers
    for i, layer in enumerate(reversed(net)):
        if isinstance(layer, nn.Flatten):
            lb = torch.reshape(lb, box_lbs.pop().shape)
            ub = torch.reshape(ub, box_ubs.pop().shape)
            # don't need to check post-condition here
        elif isinstance(layer, nn.ReLU):
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
        elif isinstance(layer, nn.Linear):
            temp_lb = layer.weight @ box_lbs.pop().squeeze(0) + layer.bias
            temp_ub = layer.weight @ box_ubs.pop().squeeze(0) + layer.bias
        elif isinstance(layer, nn.Conv2d):
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')

    return int(verified)


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
