import argparse
import torch
import torch.nn as nn

import time

from networks import get_network
from utils.loading import parse_spec
from symbolic_transforms import *
from box_transforms import propagate_linear_box, propagate_conv2d_box

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

    # set weights to no_grad:
    for param in net.parameters():
        param.requires_grad = False

    init_lb = torch.clamp(inputs - eps, 0, 1)
    init_ub = torch.clamp(inputs + eps, 0, 1)

    ub_rel = torch.eye(init_lb.numel())
    ub_rel = torch.reshape(ub_rel, (init_lb.shape[0], init_lb.shape[1], init_lb.shape[2], init_lb.shape[1] * init_lb.shape[2]))
    # add constant 0 for bias
    shape = ub_rel.shape[:-1] + (1,)
    ub_rel = torch.cat((ub_rel, torch.zeros(shape)), dim=-1)
    lb_rel = ub_rel.detach().clone()

    alphas = torch.empty(0)
    alphas.requires_grad = True
    neg_slopes = torch.empty(0)

    optimizer = torch.optim.Adam([alphas], lr=0.01)

    # propagate box through network
    for layer in net:
        if isinstance(layer, nn.Flatten):
            ub_rel = torch.flatten(ub_rel, start_dim=0, end_dim=-2)
            lb_rel = torch.flatten(lb_rel, start_dim=0, end_dim=-2)
        elif isinstance(layer, nn.Linear):
            lb_rel, ub_rel = propagate_linear_rel(lb_rel, ub_rel, layer.weight, layer.bias)
        elif isinstance(layer, nn.Conv2d):
            lb_rel, ub_rel = propagate_conv2d_rel(lb_rel, ub_rel, layer)
        elif isinstance(layer, nn.ReLU):
            lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
            curr_alphas = 0.5 * torch.ones(lb.numel())
            curr_alphas.requires_grad = True
            neg_slopes = torch.cat((neg_slopes, torch.zeros(lb.numel())))
            lb_rel, ub_rel = propagate_ReLU_rel_alpha(lb_rel, ub_rel, lb, ub, neg_slope=0, alpha=curr_alphas)
            alphas = torch.cat((alphas, curr_alphas))
        elif isinstance(layer, nn.LeakyReLU):
            lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
            curr_alphas = layer.negative_slope * torch.ones(lb.numel())
            curr_alphas.requires_grad = True
            neg_slopes = torch.cat((neg_slopes, layer.negative_slope * torch.ones(lb.numel())))
            lb_rel, ub_rel = propagate_ReLU_rel_alpha(lb_rel, ub_rel, lb, ub, neg_slope=layer.negative_slope, alpha=curr_alphas)
            alphas = torch.cat((alphas, curr_alphas))
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')

    # CHECK CONDITION
    lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)

    # GRADIENT DESCENT
    optimizer.zero_grad()  # Zero the gradients
    loss = -output_diff_lb2ub(lb, ub, true_label) + ub.max()
    loss.backward()  # Compute the gradients
    # print first 10 gradients
    print("alphas.grad: ", alphas.grad[:10])

    assert False

    optimizer.step()  # Update x

    print("dDiff target_lb, ub: ",output_diff_lb2ub(lb, ub, true_label))
    ub[true_label] = -float("inf")
    return lb[true_label] > ub.max()


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
