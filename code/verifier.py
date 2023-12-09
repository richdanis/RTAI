import argparse
import torch
import torch.nn as nn

import time

from networks import get_network
from utils.loading import parse_spec
from transforms import *

DEVICE = "cpu"


def output_diff_lb2ub(lb, ub, tue_label):
    "return difference between lower bound of target and max upper bound of other classes"
    ub[tue_label] = -float("inf")
    return float(lb[tue_label] - ub.max())


def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    
    # input shape:
    # mnist (1, 28, 28) -> x_1 , ..., x_784
    # cifar10 (3, 32, 32) -> 1_1, ... , x_3072

    # SETUP

    # set weights to no_grad:
    for param in net.parameters():
        param.requires_grad = False

    init_lb = torch.clamp(inputs - eps, 0, 1)
    init_ub = torch.clamp(inputs + eps, 0, 1)

    ub_rel = torch.eye(init_lb.numel())
    ub_rel = torch.reshape(ub_rel, (init_lb.shape[0], init_lb.shape[1], init_lb.shape[2], init_lb.shape[0] * init_lb.shape[1] * init_lb.shape[2]))
    # add constant 0 for bias
    shape = ub_rel.shape[:-1] + (1,)
    ub_rel = torch.cat((ub_rel, torch.zeros(shape)), dim=-1)
    lb_rel = ub_rel.detach().clone()

    # FORWARD PASS

    # propagate box through network
    for layer in net:
        if isinstance(layer, nn.Flatten):
            ub_rel = torch.flatten(ub_rel, start_dim=0, end_dim=-2)
            lb_rel = torch.flatten(lb_rel, start_dim=0, end_dim=-2)
        elif isinstance(layer, nn.Linear):
            lb_rel, ub_rel = transform_linear(lb_rel, ub_rel, layer.weight, layer.bias)
        elif isinstance(layer, nn.Conv2d):
            lb_rel, ub_rel = transform_conv2d(lb_rel, ub_rel, layer)
        elif isinstance(layer, nn.ReLU):
            in_shape = lb_rel.shape
            lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
            assert torch.all(lb <= ub)
            lb_rel, ub_rel = transform_ReLU(lb_rel, ub_rel, lb, ub)
            assert in_shape == lb_rel.shape
        elif isinstance(layer, nn.LeakyReLU):
            in_shape = lb_rel.shape
            lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
            assert torch.all(lb <= ub)
            lb_rel, ub_rel = transform_leakyReLU(lb_rel, ub_rel, lb, ub, slope=layer.negative_slope)
            assert in_shape == lb_rel.shape
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
        
    #checck verification old school
    lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
    lb_true = lb[true_label]
    ub[true_label] = -float("inf")
    print("differece between lb and max ub of other classes: ", float(lb_true - ub.max()))
    return int((lb_true >= ub.max()))

    # # CHECK VERIFICATION
    # init_lb = torch.flatten(init_lb)
    # init_ub = torch.flatten(init_ub)


    # differences = torch.empty((0, lb_rel.shape[-1]))
    # for i in range(10):
    #     if i == true_label:
    #         continue
    #     curr_diff = lb_rel[true_label] - ub_rel[i]
    #     curr_diff = curr_diff.unsqueeze(0)
    #     differences = torch.cat((differences, curr_diff), dim=0)

    # assert differences.shape[0] == 9

    # # lower bounds of differences must be positive
    # numerical_diff = torch.empty((0,))
    # for i in range(9):
    #     lb_temp = init_lb.clone()
    #     row = differences[i]
    #     bias = row[-1]
    #     row = row[:-1]

    #     lb_temp[row < 0] = init_ub[row < 0]

    #     diff_num = torch.sum(row * lb_temp) + bias

    #     numerical_diff = torch.cat((numerical_diff, diff_num.unsqueeze(0)), dim=0)

    # print("differece between lb and max ub of other classes: ", numerical_diff.min())

    
    # return int(numerical_diff.min() >= 0)


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
