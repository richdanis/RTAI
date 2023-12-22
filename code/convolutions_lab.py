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

    return fc, out_shape



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


def transform_conv2d(conv, shape):

        out_channels = conv.weight.shape[0]
        kernel_size = conv.weight.shape[2]
        stride = conv.stride[0]
        padding = conv.padding[0]

        # compute output shape
        out_height = (shape[1] + 2 * padding - kernel_size) // stride + 1
        out_width = (shape[2] + 2 * padding - kernel_size) // stride + 1
        out_shape = (out_channels, out_height, out_width)

        # create input matrix
        inputs = torch.eye(shape[0] * shape[1] * shape[2])

        # create index matrix
        # need to begin indexing at 1 to cope with padding
        index = torch.arange(1, inputs.shape[0] + 1, dtype=torch.float).view(shape)

        # unfold index
        index = nn.functional.unfold(index, kernel_size, padding=padding, stride=stride)
        # round index to nearest integer
        index = torch.round(index)
        index = index.long()

        # unfold input
        # prepend zeros to input to deal with padding
        inputs = torch.cat((torch.zeros(1,inputs.shape[1]), inputs), dim=0)
        inputs = inputs[index]

        # flatten weights
        weights = conv.weight.view(out_channels, -1)
        bias = conv.bias

        # compute matrix
        mat = torch.einsum('ij,jak->iak', weights, inputs)
        bias = bias.unsqueeze(1)
        bias = bias.expand(-1, mat.shape[1]).contiguous()
        bias = bias.view(-1)
        
        mat = mat.view(-1, mat.shape[-1])

        return mat, bias , out_shape



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

    input_sizes = [(1,28, 28)]

    for i in range(2):
        conv = net[i]
        
        
        img = image.clone()
        if i==1:
            img = net[0](img)
        
        fc, out_shape = torch_conv_layer_to_affine(conv, input_sizes[-1][1:])
        #fc1 = torch_conv_layer_to_affine_gpt(conv, input_sizes[-1][1:])
        #mat, bias, out_shape = transform_conv2d(conv, input_sizes[-1])


        mat1, bias1, out_shape1 = transform_conv2d(conv, input_sizes[-1])
        input_sizes.append(out_shape)
       
        res1 = (mat1@img.reshape((-1)) + bias1).reshape(conv(img).shape)
        res2 = conv(img)
        worst_error = (res1 - res2).max()

        print("Output shape", res2.shape, "Worst error: ", float(worst_error))
        assert worst_error <= 1.0e-6


    
    




    

    return 0






if __name__ == "__main__":
    main()
