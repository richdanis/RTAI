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

    print(net)

    # set weights to no_grad:
    for param in net.parameters():
        param.requires_grad = False

    init_lb = torch.clamp(inputs - eps, 0, 1)
    init_ub = torch.clamp(inputs + eps, 0, 1)

    alphas = None
    neg_slopes = torch.empty(0)

    verified = False
    first_pass = True

    optimizer = None

    while not verified:

        ub_rel = torch.eye(init_lb.numel())
        ub_rel = torch.reshape(ub_rel, (init_lb.shape[0], init_lb.shape[1], init_lb.shape[2], init_lb.shape[0] * init_lb.shape[1] * init_lb.shape[2]))
        # add constant 0 for bias
        shape = ub_rel.shape[:-1] + (1,)
        ub_rel = torch.cat((ub_rel, torch.zeros(shape)), dim=-1)
        lb_rel = ub_rel.detach().clone()

        c = 0

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
                # lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
                # neg_slopes = torch.cat((neg_slopes, torch.zeros(lb.numel())))
                # if alphas is None:
                #     alphas = 1.0 * torch.ones(lb.numel(), requires_grad=True)
                # elif first_pass:
                #     alphas = torch.cat((alphas, 1.0 * torch.ones(lb.numel(), requires_grad=True)))
                # lb_rel, ub_rel = propagate_ReLU_rel_alpha(lb_rel, ub_rel, lb, ub, alpha=alphas[c:c+lb.numel()])
                # c += lb.numel()
                # approximate as identity
                assert in_shape == lb_rel.shape
            elif isinstance(layer, nn.LeakyReLU):
                in_shape = lb_rel.shape
                lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
                # if alphas is None:
                #     alphas = layer.negative_slope * torch.ones(lb.numel(), requires_grad=True)
                # elif first_pass:
                #     alphas = torch.cat((alphas, layer.negative_slope * torch.ones(lb.numel(), requires_grad=True)))
                neg_slopes = torch.cat((neg_slopes, layer.negative_slope * torch.ones(lb.numel())))
                lb_rel, ub_rel = transform_leakyReLU_alpha(lb_rel, ub_rel, lb, ub, slope=layer.negative_slope)
                # c += lb.numel()
                assert in_shape == lb_rel.shape
            else:
                raise NotImplementedError(f'Unsupported layer type: {type(layer)}')

        if alphas is None:
            lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
            ub[true_label] = -float("inf")
            verified =  lb[true_label] > ub.max()

            return int(verified)


        alphas.retain_grad()

        if first_pass:
            # vllt anderen optimizer nehmen
            optimizer = torch.optim.Adam([alphas], lr=10)
        optimizer.zero_grad()
        lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
        loss = - lb[true_label] + sum(ub[ub_label] for ub_label in range(10) if ub_label != true_label)
        loss.backward(retain_graph=True)

        optimizer.step()

        # clamp alphas back to [0, inf]
        alphas.data = torch.clamp(alphas.data, 0, float("inf"))

        # free computational graph

        # print("dDiff target_lb, ub: ",output_diff_lb2ub(lb, ub, true_label))
        ub[true_label] = -float("inf")
        verified =  lb[true_label] > ub.max()

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
