import argparse
import torch
import torch.nn as nn

import time

from networks import get_network
from utils.loading import parse_spec
from symbolic_transforms import propagate_linear_symbolic, propagate_conv2d_symbolic, propagate_linear_rel, propagate_conv2d_rel, evaluate_bounds, propagate_ReLU_rel
from box_transforms import propagate_linear_box, propagate_conv2d_box

DEVICE = "cpu"


def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    
    # input shape:
    # mnist (1, 28, 28) -> x_1 , ..., x_784
    # cifar10 (3, 32, 32) -> 1_1, ... , x_3072

    # n_ij <= f(x_1, ... , x_784) + b- > (785,)
    # n_ij >= f(x_1, ... , x_784) + b -> (785,)

    # i-th layer: m neurons -> (m, 785)

    # set weights to no_grad:
    for param in net.parameters():
        param.requires_grad = False

    init_lb = torch.clamp(inputs - eps, 0, 1)
    init_ub = torch.clamp(inputs + eps, 0, 1)

    # n_ij = f(x_1, ... , x_784)

    # n_ij <= f(x_1, ... , x_784) + a
    # n_ij >= f(x_1, ... , x_784) + a

    # x1 + 87 x1 -27 x2 = 88 x1 - 27 x2

    # n_ij <= ub
    # n_ij >= lb

    ub_rel = torch.eye(init_lb.numel())
    ub_rel = torch.reshape(ub_rel, (init_lb.shape[0], init_lb.shape[1], init_lb.shape[2], init_lb.shape[1] * init_lb.shape[2]))
    # add constant 0 for bias
    shape = ub_rel.shape[:-1] + (1,)
    ub_rel = torch.cat((ub_rel, torch.zeros(shape)), dim=-1)
    lb_rel = ub_rel.detach().clone()

    # lbs_box = [init_lb]
    # ubs_box = [init_ub]

    # propagate box through network
    for layer in net:
        # if sequential, list modules of the sequential
        # all entries in lbs_box must be smaller than all entries in ubs_box
        #assert torch.all(lbs_box[-1] <= ubs_box[-1])
        if isinstance(layer, nn.Flatten):
            ub_rel = torch.flatten(ub_rel, start_dim=0, end_dim=-2)
            lb_rel = torch.flatten(lb_rel, start_dim=0, end_dim=-2)
            # lbs_box.append(torch.flatten(lbs_box[-1], start_dim=0))
            # ubs_box.append(torch.flatten(ubs_box[-1], start_dim=0))
        elif isinstance(layer, nn.Linear):
            #ub_rel = propagate_linear_symbolic(ub_rel, layer.weight, layer.bias)
            lb_rel, ub_rel = propagate_linear_rel(lb_rel, ub_rel, layer.weight, layer.bias)
            # curr_lb, curr_ub = propagate_linear_box(lbs_box[-1], ubs_box[-1], layer)
            # lbs_box.append(curr_lb)
            # ubs_box.append(curr_ub)
        elif isinstance(layer, nn.Conv2d):
            #ub_rel = propagate_conv2d_symbolic(ub_rel, layer)
            lb_rel, ub_rel = propagate_conv2d_rel(lb_rel, ub_rel, layer)
            # curr_lb, curr_ub = propagate_conv2d_box(lbs_box[-1], ubs_box[-1], layer)
            # lbs_box.append(curr_lb)
            # ubs_box.append(curr_ub)
        elif isinstance(layer, nn.ReLU):
            # treat as identity (bad over approximation)
            lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
            lb_rel, ub_rel = propagate_ReLU_rel(lb_rel, ub_rel, lb, ub)
            # ubs_box.append(layer(ubs_box[-1]))
            # lbs_box.append(layer(lbs_box[-1]))
        elif isinstance(layer, nn.LeakyReLU):
            # treat as identity (bad over approximation)
            # ubs_box.append(layer(ubs_box[-1]))
            # lbs_box.append(layer(lbs_box[-1]))
            pass
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')

    # CHECK CONDITION

    lb, ub = evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel)
    ub[true_label] = -float("inf")
    return lb[true_label] > ub.max()

    # init_lb = init_lb.flatten()
    # init_ub = init_ub.flatten()

    # # at the end ub_rel is always of shape (10, num_symbols + 1)
    # # for mnist (10, 785)

    # # w_1 x_1 + ... + w_784 x_784 + b
    # # f(alpha_1, alpha_2, ...)
    # # 10 output classes
    # # target class: lowerbound > upperbound of other classes
    # # target: lower bound maximieren
    # # other: upper bound minimieren

    # ub_out = torch.zeros(ub_rel.shape[0])
    # lb_out = torch.zeros(ub_rel.shape[0])
    # # given input matrix and input bounds compute output bounds
    # for i in range(ub_rel.shape[0]):
    #     row = ub_rel[i,:-1]
    #     b = ub_rel[i,-1]
    #     ub_temp = init_ub.detach().clone()
    #     ub_temp[row < 0] = init_lb[row < 0]
    #     lb_temp = init_lb.detach().clone()
    #     lb_temp[row < 0] = init_ub[row < 0]
    #     ub_out[i] = row @ ub_temp + b
    #     lb_out[i] = row @ lb_temp + b

    # # check post-condition
    # ub_out[true_label] = -float("inf")
    # return lb_out[true_label] > ub_out.max()


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
