import argparse
import torch
import torch.nn as nn

import time

from networks import get_network
from utils.loading import parse_spec
from transforms import *

DEVICE = "cpu"

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

    alphas = []
    optimizer = None
    first_pass = True
    verified = False
    k = 0

    while not verified:

        k += 1

        ub_rel = torch.eye(init_lb.numel())
        ub_rel = ub_rel.view((init_lb.shape[0], init_lb.shape[1], init_lb.shape[2], init_lb.shape[0] * init_lb.shape[1] * init_lb.shape[2]))
        # add constant 0 for bias
        shape = ub_rel.shape[:-1] + (1,)
        ub_rel = torch.cat((ub_rel, torch.zeros(shape)), dim=-1)
        lb_rel = ub_rel.detach().clone()

        # FORWARD PASS
        i = 0
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
                lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
                if first_pass:
                    alphas = []
                    alpha = torch.where(ub > lb.abs(), torch.ones_like(lb), torch.zeros_like(lb))
                    alpha.requires_grad = True
                    alphas.append(alpha)
                lb_rel, ub_rel = transform_ReLU_alpha(lb_rel, ub_rel, lb, ub, alphas[i])
                i += 1
            elif isinstance(layer, nn.LeakyReLU):
                lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
                lb_rel, ub_rel = transform_leakyReLU(lb_rel, ub_rel, lb, ub, slope=layer.negative_slope)
            else:
                raise NotImplementedError(f'Unsupported layer type: {type(layer)}')

        # CHECK VERIFICATION
        lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
        diffs = differences(init_lb, init_ub, lb_rel, ub_rel, true_label)
        verified = diffs.min() >= 0

        print("Diffs: ", diffs)
        # relu on diffs
        # maybe add additional loss term
        loss = torch.relu(- diffs).sum().log()

        # BACKWARD PASS
        if first_pass:
            optimizer = torch.optim.Adam(alphas, lr=1)
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        first_pass = False

    return verified


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
