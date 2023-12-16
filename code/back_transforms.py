import torch
import torch.nn as nn

class Bounds():

        def __init__(self, lb_mat, ub_mat, lb_bias, ub_bias):
                
                self.lb_mat = lb_mat
                self.ub_mat = ub_mat

                self.lb_bias = lb_bias
                self.ub_bias = ub_bias

        def back(self, curr_bound):

                curr_ub = curr_bound.ub_mat
                curr_ub_bias = curr_bound.ub_bias

                pos_ub = torch.where(curr_ub > 0, curr_ub, torch.zeros_like(curr_ub))
                neg_ub = torch.where(curr_ub < 0, curr_ub, torch.zeros_like(curr_ub))

                res_ub = pos_ub @ self.ub_mat + neg_ub @ self.lb_mat
                res_ub_bias = pos_ub @ self.ub_bias + neg_ub @ self.lb_bias + curr_ub_bias

                curr_lb = curr_bound.lb_mat
                curr_lb_bias = curr_bound.lb_bias

                pos_lb = torch.where(curr_lb > 0, curr_lb, torch.zeros_like(curr_lb))
                neg_lb = torch.where(curr_lb < 0, curr_lb, torch.zeros_like(curr_lb))

                res_lb = pos_lb @ self.lb_mat + neg_lb @ self.ub_mat
                res_lb_bias = pos_lb @ self.lb_bias + neg_lb @ self.ub_bias + curr_lb_bias

                return Bounds(res_lb, res_ub, res_lb_bias, res_ub_bias)
        
class ReLU_Bounds():

        def __init__(self, lb, ub, neg_slope, alpha):

                self.neg_slope = neg_slope
                self.alpha = alpha
                self.lb = lb
                self.ub = ub

        def get_bounds(self):

                # clamp back alpha
                with torch.no_grad():
                        self.alpha.clamp_(self.neg_slope, 1)

                # masks
                crossing = torch.logical_and(self.ub > 0, self.lb < 0)
                pos = self.lb > 0
                neg = self.ub <= 0

                # default weights
                ones = torch.where(pos, torch.ones_like(self.lb), torch.zeros_like(self.lb))
                neg_ones = torch.where(neg, self.neg_slope * torch.ones_like(self.lb), torch.zeros_like(self.lb))

                if self.neg_slope < 1.0:

                        # compute upper slope and bias
                        upper_slope = torch.where(crossing, (self.ub - self.neg_slope*self.lb) / (self.ub - self.lb), torch.zeros_like(self.lb))
                        weight_ub = torch.diag(ones + neg_ones + upper_slope)
                        bias_ub = torch.where(crossing, self.ub * self.lb * (self.neg_slope - 1) / (self.ub - self.lb), torch.zeros_like(self.lb))

                        # compute lower slope and bias
                        lower_slope = torch.where(crossing, self.alpha, torch.zeros_like(self.lb))
                        weight_lb = torch.diag(ones + neg_ones + lower_slope)
                        bias_lb = torch.zeros_like(self.lb)

                        return Bounds(weight_lb, weight_ub, bias_lb, bias_ub)

                elif self.neg_slope > 1.0:

                        # compute lower slope and bias
                        lower_slope = torch.where(crossing, (self.ub - self.neg_slope*self.lb) / (self.ub - self.lb), torch.zeros_like(self.lb))
                        weight_lb = torch.diag(ones + neg_ones + lower_slope)
                        bias_lb = torch.where(crossing, self.ub * self.lb * (self.neg_slope - 1) / (self.ub - self.lb), torch.zeros_like(self.lb))

                        # compute upper slope and bias
                        upper_slope = torch.where(crossing, self.alpha, torch.zeros_like(self.lb))
                        weight_ub = torch.diag(ones + neg_ones + upper_slope)
                        bias_ub = torch.zeros_like(self.lb)

                        return Bounds(weight_lb, weight_ub, bias_lb, bias_ub)

        def back(self, curr_bound):

                prev_bound = self.get_bounds()

                return prev_bound.back(curr_bound)


def transform_linear(weight, bias):

        return Bounds(weight, weight, bias, bias)

def transform_ReLU_alpha(lb, ub, alpha):
        
        # careful might want to flatten

        crossing = torch.logical_and(ub > 0, lb < 0)
        pos = lb > 0

        ones = torch.where(pos, torch.ones_like(lb), torch.zeros_like(lb))

        upper_slope = torch.where(crossing, ub / (ub - lb), torch.zeros_like(lb))
        lower_slope = torch.where(crossing, alpha, torch.zeros_like(lb))
        
        bias_ub = torch.where(crossing, -ub * lb / (ub - lb), torch.zeros_like(lb))
        bias_lb = torch.zeros_like(lb)

        weight_lb = torch.diag(ones + lower_slope)
        weight_ub = torch.diag(ones + upper_slope)

        return Bounds(weight_lb, weight_ub, bias_lb, bias_ub)

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
        index = torch.arange(inputs.shape[0], dtype=torch.float).view(shape)

        # unfold index
        index = nn.functional.unfold(index, kernel_size, padding=padding, stride=stride)
        index = index.long()

        # unfold input
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

        return Bounds(mat, mat, bias, bias), out_shape

def transform_leakyReLU(lb, ub, slope, alpha):

        # masks
        crossing = torch.logical_and(ub > 0, lb < 0)
        pos = lb > 0
        neg = ub <= 0

        # default weights
        ones = torch.where(pos, torch.ones_like(lb), torch.zeros_like(lb))
        neg_ones = torch.where(neg, slope * torch.ones_like(lb), torch.zeros_like(lb))

        if slope < 1.0:

                # compute upper slope and bias
                upper_slope = torch.where(crossing, (ub - slope*lb) / (ub - lb), torch.zeros_like(lb))
                weight_ub = torch.diag(ones + neg_ones + upper_slope)
                bias_ub = torch.where(crossing, ub * lb * (slope - 1) / (ub - lb), torch.zeros_like(lb))

                # compute lower slope and bias
                lower_slope = torch.where(crossing, alpha, torch.zeros_like(lb))
                weight_lb = torch.diag(ones + neg_ones + lower_slope)
                bias_lb = torch.zeros_like(lb)

                return Bounds(weight_lb, weight_ub, bias_lb, bias_ub)

        elif slope > 1.0:

                # compute lower slope and bias
                lower_slope = torch.where(crossing, (ub - slope*lb) / (ub - lb), torch.zeros_like(lb))
                weight_lb = torch.diag(ones + neg_ones + lower_slope)
                bias_lb = torch.where(crossing, ub * lb * (slope - 1) / (ub - lb), torch.zeros_like(lb))

                # compute upper slope and bias
                upper_slope = torch.where(crossing, alpha, torch.zeros_like(lb))
                weight_ub = torch.diag(ones + neg_ones + upper_slope)
                bias_ub = torch.zeros_like(lb)

                return Bounds(weight_lb, weight_ub, bias_lb, bias_ub)

def back(bounds):

        curr_bound = bounds.pop()

        while bounds:
                curr_bound = bounds.pop().back(curr_bound)

        return curr_bound

def eval(bound, init_lb, init_ub):

        pos_ub = torch.where(bound.ub_mat > 0, bound.ub_mat, torch.zeros_like(bound.ub_mat))
        neg_ub = torch.where(bound.ub_mat < 0, bound.ub_mat, torch.zeros_like(bound.ub_mat))

        pos_lb = torch.where(bound.lb_mat > 0, bound.lb_mat, torch.zeros_like(bound.lb_mat))
        neg_lb = torch.where(bound.lb_mat < 0, bound.lb_mat, torch.zeros_like(bound.lb_mat))

        ub = pos_ub @ init_ub + neg_ub @ init_lb + bound.ub_bias
        lb = pos_lb @ init_lb + neg_lb @ init_ub + bound.lb_bias

        return lb, ub