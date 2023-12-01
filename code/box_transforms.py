import torch
import torch.nn as nn

def propagate_linear_box(lb, ub, fc: nn.Linear):
        """
        Propagate an abstract box through a linear layer.
        """
        n_neurons = fc.weight.shape[0]

        ub_temp = ub.repeat(n_neurons,1)
        lb_temp = lb.repeat(n_neurons,1)

        ub_temp[fc.weight < 0] = lb_temp[fc.weight < 0]
        lb_temp[fc.weight < 0] = ub.repeat(n_neurons,1)[fc.weight < 0]

        ub_res = (fc.weight * ub_temp).sum(dim=1) + fc.bias
        lb_res = (fc.weight * lb_temp).sum(dim=1) + fc.bias

        return lb_res.unsqueeze(0), ub_res.unsqueeze(0)

def propagate_conv2d_box(lb, ub, conv: nn.Conv2d):
        """
        Propagate an abstract box through a convolutional layer.
        Bounds shape: (in_channels, height, width)
        """
        assert len(lb.shape) == 3
        # get number of channels
        out_channels = conv.weight.shape[0]
        kernel_size = conv.weight.shape[2]
        stride = conv.stride[0]
        padding = conv.padding[0]

        # compute output shape
        out_height = (lb.shape[1] + 2 * padding - kernel_size) // stride + 1
        out_width = (lb.shape[2] + 2 * padding - kernel_size) // stride + 1
        out_shape = (out_channels, out_height, out_width)

        # unfold bounds
        lb = torch.nn.functional.unfold(lb, (kernel_size, kernel_size), stride=stride, padding=padding)
        ub = torch.nn.functional.unfold(ub, (kernel_size, kernel_size), stride=stride, padding=padding)

        # get weight and bias
        w = conv.weight
        w = w.view(w.shape[0], -1)
        # w is now of shape (out_channels, in_channels * kernel_size * kernel_size)
        b = conv.bias
        # b is of shape (out_channels,)

        # pass weight, bias and unfolded input through linear layer
        # issue here is that we have a matrix matrix multiplication and not a matrix vector multiplication
        lb, ub = matrix_matrix_mul_box(lb, ub, w, b)
        assert len(lb.shape) == 2
        # reshape to output shape
        lb = lb.view(out_shape)
        ub = ub.view(out_shape)

        return lb, ub

def matrix_matrix_mul_box(lb, ub, weight, bias):
        """
        Propagate an abstract box through a matrix multiplication.
        """
        lb_res = torch.zeros(weight.shape[0], lb.shape[1])
        ub_res = torch.zeros(weight.shape[0], lb.shape[1])
        
        for i in range(weight.shape[0]):
                weight_row = weight[i]
                for j in range(lb.shape[1]):
                        lb_temp = lb[:,j].detach().clone()
                        ub_temp = ub[:,j].detach().clone()
                        lb_temp[weight_row < 0] = ub[weight_row < 0, j]
                        ub_temp[weight_row < 0] = lb[weight_row < 0, j]
                        lb_res[i,j] = weight_row @ lb_temp + bias[i]
                        ub_res[i,j] = weight_row @ ub_temp + bias[i]
        return lb_res, ub_res