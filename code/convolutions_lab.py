import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch
import torch.nn as nn

import time

from networks import get_network
from utils.loading import parse_spec
from transforms import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
import copy


def torch_conv_layer_to_affine(
    conv: torch.nn.Conv2d, input_size: Tuple[int, int]
) -> torch.nn.Linear:
    w, h = input_size

    # Formula from the Torch docs:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    output_size = [
        (input_size[i] + 2 * conv.padding[i] - conv.kernel_size[i]) // conv.stride[i]
        + 1
        for i in [0, 1]
    ]

    in_shape = (conv.in_channels, w, h)
    out_shape = (conv.out_channels, output_size[0], output_size[1])

    fc = nn.Linear(in_features=np.product(in_shape), out_features=np.product(out_shape))
    fc.weight.data.fill_(0.0)

    fc.weight.data = fc.weight.data.clone()
    fc.bias.data = fc.bias.data.clone()


    # Output coordinates
    for xo, yo in range2d(output_size[0], output_size[1]):
        # The upper-left corner of the filter in the input tensor
        xi0 = -conv.padding[0] + conv.stride[0] * xo
        yi0 = -conv.padding[1] + conv.stride[1] * yo

        # Position within the filter
        for xd, yd in range2d(conv.kernel_size[0], conv.kernel_size[1]):
            # Output channel
            for co in range(conv.out_channels):
                fc.bias.data[enc_tuple((co, xo, yo), out_shape)] = conv.bias[co]
                for ci in range(conv.in_channels):
                    # Make sure we are within the input image (and not in the padding)
                    if 0 <= xi0 + xd < w and 0 <= yi0 + yd < h:
                        cw = conv.weight[co, ci, xd, yd]
                        # Flatten the weight position to 1d in "canonical ordering",
                        # i.e. guaranteeing that:
                        # FC(img.reshape(-1)) == Conv(img).reshape(-1)
                        fc.weight.data[
                            enc_tuple((co, xo, yo), out_shape),
                            enc_tuple((ci, xi0 + xd, yi0 + yd), in_shape),
                        ] = cw

    return fc


def range2d(to_a, to_b):
    for a in range(to_a):
        for b in range(to_b):
            yield a, b


def enc_tuple(tup: Tuple, shape: Tuple) -> int:
    res = 0
    coef = 1
    for i in reversed(range(len(shape))):
        assert tup[i] < shape[i]
        res += coef * tup[i]
        coef *= shape[i]

    return res


def dec_tuple(x: int, shape: Tuple) -> Tuple:
    res = []
    for i in reversed(range(len(shape))):
        res.append(x % shape[i])
        x //= shape[i]

    return tuple(reversed(res))


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

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt")

    image = image
    out = net(image.unsqueeze(0))

    inputs = image

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label


    for param in net.parameters():
       param.requires_grad = False

    #set input to no_grad:
    inputs.requires_grad = False


    conv = net[0]   
    
    # Assume the input size is (1,28, 28)
    input_size = (1,28, 28)

    #img = torch.rand((1, 2, 6, 7))
    img = image.clone()
    fc = torch_conv_layer_to_affine(conv, img.shape[1:])

    # Also checks that our encoding flattens the inputs/outputs such that
    # FC(flatten(img)) == flatten(Conv(img))
    res1 = fc(img.reshape((-1))).reshape(conv(img).shape)
    res2 = conv(img)
    worst_error = (res1 - res2).max()

    print("Output shape", res2.shape, "Worst error: ", float(worst_error))
    assert worst_error <= 1.0e-6
    

    return 0






if __name__ == "__main__":
    main()
