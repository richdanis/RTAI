import torch
import torch.nn as nn

class Bounds():

        def __init__(self, lb_mat, ub_mat, lb_bias, ub_bias):

                # bounds for each layer
                
                self.lb_mat = lb_mat
                self.ub_mat = ub_mat

                self.lb_bias = lb_bias
                self.ub_bias = ub_bias

        def back(self, curr_bound):

                # backsubstitute curr_bound through layer

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

                # special class for ReLUs as we need to update
                # the alpha parameter during backsubstitution

                self.neg_slope = neg_slope
                self.alpha = alpha
                self.lb = lb
                self.ub = ub
                self.lb_mat = None
                self.ub_mat = None
                self.lb_bias = None
                self.ub_bias = None

        def get_bounds(self):

                # clamp back alpha to valid range
                with torch.no_grad():
                        if self.neg_slope < 1.0:
                                self.alpha.clamp_(self.neg_slope, 1)
                        else:
                                self.alpha.clamp_(1, self.neg_slope)

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

                        #return Bounds(weight_lb, weight_ub, bias_lb, bias_ub)
                        self.lb_mat = weight_lb
                        self.ub_mat = weight_ub

                        self.lb_bias = bias_lb
                        self.ub_bias = bias_ub

                elif self.neg_slope > 1.0:

                        # compute lower slope and bias
                        lower_slope = torch.where(crossing, (self.ub - self.neg_slope*self.lb) / (self.ub - self.lb), torch.zeros_like(self.lb))
                        weight_lb = torch.diag(ones + neg_ones + lower_slope)
                        bias_lb = torch.where(crossing, self.ub * self.lb * (self.neg_slope - 1) / (self.ub - self.lb), torch.zeros_like(self.lb))

                        # compute upper slope and bias
                        upper_slope = torch.where(crossing, self.alpha, torch.zeros_like(self.lb))
                        weight_ub = torch.diag(ones + neg_ones + upper_slope)
                        bias_ub = torch.zeros_like(self.lb)

                        #return Bounds(weight_lb, weight_ub, bias_lb, bias_ub)
                        self.lb_mat = weight_lb
                        self.ub_mat = weight_ub

                        self.lb_bias = bias_lb
                        self.ub_bias = bias_ub

        def back(self, curr_bound):

                # get new bounds using updated alpha

                #prev_bound = self.get_bounds()

                self.get_bounds()

                # perform backsubstitution

                #return prev_bound.back(curr_bound)
                return Bounds(self.lb_mat, self.ub_mat, self.lb_bias, self.ub_bias).back(curr_bound)

def transform_linear(weight, bias):

        return Bounds(weight, weight, bias, bias)

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
        # round index to nearest integer
        index = torch.round(index)
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

def update_ReLUs(bounds, init_lb, init_ub):

        # update ReLU bounds

        for i in range(len(bounds)):

                if isinstance(bounds[i], ReLU_Bounds):

                        # get new lb, ub
                        curr_bounds = back(bounds[:i].copy())
                        lb, ub = eval(curr_bounds, init_lb, init_ub)

                        # update ReLU
                        bounds[i].lb = lb
                        bounds[i].ub = ub
                        
                        # update bounds
                        bounds[i].get_bounds()

def back(bounds):

        # backsubstitute bounds through network

        curr_bound = bounds.pop()

        while bounds:

                curr_bound = bounds.pop().back(curr_bound)

        return curr_bound

def eval(bound, init_lb, init_ub):

        # evaluate bounds on initial box

        pos_ub = torch.where(bound.ub_mat > 0, bound.ub_mat, torch.zeros_like(bound.ub_mat))
        neg_ub = torch.where(bound.ub_mat < 0, bound.ub_mat, torch.zeros_like(bound.ub_mat))

        pos_lb = torch.where(bound.lb_mat > 0, bound.lb_mat, torch.zeros_like(bound.lb_mat))
        neg_lb = torch.where(bound.lb_mat < 0, bound.lb_mat, torch.zeros_like(bound.lb_mat))

        ub = pos_ub @ init_ub + neg_ub @ init_lb + bound.ub_bias
        lb = pos_lb @ init_lb + neg_lb @ init_ub + bound.lb_bias

        return lb, ub