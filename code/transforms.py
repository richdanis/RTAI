import torch
import torch.nn as nn


def transform_linear(lb_rel, ub_rel, weight, bias):
        """
        Propagate relational bounds through a linear layer.
        """
        lb_res = torch.empty(weight.shape[0], lb_rel.shape[-1])
        ub_res = torch.empty(weight.shape[0], lb_rel.shape[-1])
        for i in range(weight.shape[0]):

               row = weight[i,:]

               lb_temp = lb_rel.clone()
               ub_temp = ub_rel.clone()

               lb_temp[row < 0,:] = ub_rel[row < 0,:]
               ub_temp[row < 0,:] = lb_rel[row < 0,:]

               row = row.unsqueeze(1)

               lb_temp = row * lb_temp
               ub_temp = row * ub_temp

               lb_temp = lb_temp.sum(dim=0)
               ub_temp = ub_temp.sum(dim=0)

               lb_temp[-1] = lb_temp[-1] + bias[i]
               ub_temp[-1] = ub_temp[-1] + bias[i]

               lb_res[i] = lb_temp
               ub_res[i] = ub_temp
               
        return lb_res, ub_res

def matrix_matrix_mul_rel(lb_rel, ub_rel, weight, bias):
        """
        Propagate an abstract box through a matrix multiplication.
        """
        lb_res = torch.empty(weight.shape[0], lb_rel.shape[1], lb_rel.shape[2])
        ub_res = torch.empty(weight.shape[0], lb_rel.shape[1], lb_rel.shape[2])
        for i in range(weight.shape[0]):
                row = weight[i,:]
                for j in range(lb_rel.shape[1]):
                        # multiply each column of weight matrix with each row of inputs
                        lb_temp = lb_rel[:,j,:].clone()
                        ub_temp = ub_rel[:,j,:].clone()

                        lb_temp[row < 0,:] = ub_temp[row < 0,:]
                        ub_temp[row < 0,:] = lb_rel[:,j,:][row < 0,:]

                        lb_temp = row.unsqueeze(1) * lb_temp
                        ub_temp = row.unsqueeze(1) * ub_temp

                        lb_temp = lb_temp.sum(dim=0)
                        ub_temp = ub_temp.sum(dim=0)
                        # sum over rows
                        lb_temp[-1] = lb_temp[-1] + bias[i]
                        ub_temp[-1] = ub_temp[-1] + bias[i]
                        #
                        lb_res[i,j] = lb_temp
                        ub_res[i,j] = ub_temp
        return lb_res, ub_res

def transform_conv2d(lb_rel, ub_rel, conv: nn.Conv2d):
        """
        Propagate relational bounds through a convolutional layer.
        lb_rel shape: (in_channels, height, width, number_of_symbols)
        """
        assert len(lb_rel.shape) == 4
        # get number of channels
        out_channels = conv.weight.shape[0]
        kernel_size = conv.weight.shape[2]
        stride = conv.stride[0]
        padding = conv.padding[0]

        # compute output shape
        out_height = (lb_rel.shape[1] + 2 * padding - kernel_size) // stride + 1
        out_width = (lb_rel.shape[2] + 2 * padding - kernel_size) // stride + 1
        out_shape = (out_channels, out_height, out_width, lb_rel.shape[-1])

        # index array
        shape = torch.tensor(lb_rel.shape)
        num_ind = shape[:-1].prod()
        ind = torch.arange(0, num_ind, dtype=torch.float)
        ind = ind.reshape(lb_rel.shape[:-1])

        # unfold index array
        ind = torch.nn.functional.unfold(ind, (kernel_size, kernel_size), stride=stride, padding=padding)
        # change to int
        ind = ind.int()

        # flatten input
        lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)
        ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)

        assert len(ind.shape) == 2
        # ind is now of shape (in_channels * kernel_size * kernel_size, num_patches)
        # unfold input

        lb_unfold = torch.empty(ind.shape + (lb_rel.shape[-1],))
        ub_unfold = torch.empty(ind.shape + (ub_rel.shape[-1],))
        for i in range(ind.shape[0]):
                for j in range(ind.shape[1]):
                        lb_unfold[i, j] = lb_rel[ind[i, j]]
                        ub_unfold[i, j] = ub_rel[ind[i, j]]

        # get weight and bias
        w = conv.weight
        w = w.view(w.shape[0], -1)
        # w is now of shape (out_channels, in_channels * kernel_size * kernel_size)
        b = conv.bias
        # b is of shape (out_channels,)

        # pass weight, bias and lb_unfold input through linear layer
        # issue here is that we have a matrix matrix multiplication and not a matrix vector multiplication
        lb_res, ub_res = matrix_matrix_mul_rel(lb_unfold, ub_unfold, w, b)
        assert len(lb_res.shape) == 3
        # reshape to output shape
        lb_res = lb_res.view(out_shape)
        ub_res = ub_res.view(out_shape)

        return lb_res, ub_res

def transform_ReLU(lb_rel, ub_rel, lb, ub):
        """
        Propagate relational bounds through a ReLU layer.
        lb_rel  and ub_rel of shape = ([50, 785]) (batch_size, number_of_symbols + 1)
        lb and ub of shape = ([50]) (batch_size)

        """
        lb_rel_before = lb_rel.clone()
        ub_rel_before = ub_rel.clone()
        in_shape = lb_rel.shape
        in_shape_lb = lb.shape
        #check that lb and lb_rel have same batch size
        assert(in_shape[0]== in_shape_lb[0])
        #asssert that lb is smaller than ub
        assert(torch.all(lb <= ub))

        upper_slope = ub / (ub - lb)

        ub_res = upper_slope.unsqueeze(-1) * (ub_rel - lb.unsqueeze(-1))

        # flatten ub, ub_rel
        ub = ub.flatten(start_dim=0)
        ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)
        ub_res = ub_res.flatten(start_dim=0, end_dim=-2)

        # flatten lb, lb_rel
        lb = lb.flatten(start_dim=0)
        lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)
        lb_res = lb_rel.flatten(start_dim=0, end_dim=-2)

        lb_res[:,:] = 0

        lb_res[ub < 0,:] = 0
        ub_res[ub < 0,:] = 0

        ub_res[lb > 0,:] = ub_rel[lb > 0,:]
        lb_res[lb > 0,:] = lb_rel[lb > 0,:]

        lb_res = lb_res.view(in_shape)
        ub_res = ub_res.view(in_shape)

        return lb_res, ub_res

def transform_ReLU_alpha(lb_rel, ub_rel, lb, ub, alpha):

        upper_slope = ub / (ub - lb)

        ub_res = upper_slope.unsqueeze(-1) * (ub_rel.clone() - lb.unsqueeze(-1))

        # flatten ub, ub_rel
        ub = ub.flatten(start_dim=0)
        ub_res = ub_res.flatten(start_dim=0, end_dim=-2)

        alpha = alpha.view(lb.shape).unsqueeze(-1)
        
        lb_res = alpha * lb_rel.clone()
        # flatten lb, lb_rel
        lb = lb.flatten(start_dim=0)
        lb_res = lb_res.flatten(start_dim=0, end_dim=-2)

        ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)
        lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)
        
        # to zero if ub < 0
        ub_res[ub < 0,:] = 0
        lb_res[ub < 0,:] = 0

        # don't change if lb > 0
        ub_res[lb > 0,:] = ub_rel[lb > 0,:]
        lb_res[lb > 0,:] = lb_rel[lb > 0,:]

        ub_res = ub_res.view(ub_rel.shape)
        lb_res = lb_res.view(ub_rel.shape)

        return lb_res, ub_res


def transform_leakyReLU(lb_rel, ub_rel, lb, ub, slope, alpha = 1):

        in_shape = lb_rel.shape

        if slope == 1.0:

                return lb_rel, ub_rel
        
        elif slope < 1.0:

                # compute upper slope and offset
                upper_slope = (ub - slope*lb) / (ub - lb)
                upper_slope = upper_slope.unsqueeze(-1)

                offset = ub * lb * (slope - 1) / (ub - lb)
                offset = offset.flatten(start_dim=0)

                # default upper and lower bound
                # maybe add offset only to constant?!
                ub_res = upper_slope * ub_rel
                lb_res = slope * lb_rel

                # flatten
                ub = ub.flatten(start_dim=0)
                ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)
                ub_res = ub_res.flatten(start_dim=0, end_dim=-2)

                lb = lb.flatten(start_dim=0)
                lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)
                lb_res = lb_res.flatten(start_dim=0, end_dim=-2)

                # add offset
                ub_res[:,-1] = ub_res[:,-1] + offset
                
                # when not crossing
                ub_res[ub < 0,:] = slope * ub_rel[ub < 0,:]
                lb_res[ub < 0,:] = slope * lb_rel[ub < 0,:]

                lb_res[lb > 0,:] = lb_rel[lb > 0,:]
                ub_res[lb > 0,:] = ub_rel[lb > 0,:]
                

                return lb_res.view(in_shape), ub_res.view(in_shape)

        elif slope > 1.0:

                # compute upper slope and offset
                lower_slope = (ub - slope*lb) / (ub - lb)
                lower_slope = lower_slope.unsqueeze(-1)

                offset = ub * lb * (slope - 1) / (ub - lb)
                offset = offset.flatten(start_dim=0)
                
                # default upper and lower bound
                ub_res = slope * ub_rel
                lb_res = lower_slope * lb_rel

                # flatten
                ub = ub.flatten(start_dim=0)
                ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)
                ub_res = ub_res.flatten(start_dim=0, end_dim=-2)

                lb = lb.flatten(start_dim=0)
                lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)
                lb_res = lb_res.flatten(start_dim=0, end_dim=-2)

                # add offset
                lb_res[:,-1] = lb_res[:,-1] + offset
                
                # when not crossing
                ub_res[ub < 0,:] = slope * ub_rel[ub < 0,:]
                lb_res[ub < 0,:] = slope * lb_rel[ub < 0,:]

                lb_res[lb > 0,:] = lb_rel[lb > 0,:]
                ub_res[lb > 0,:] = ub_rel[lb > 0,:]

                return lb_res.view(in_shape), ub_res.view(in_shape)

def evaluate_bounds(init_lb, init_ub, lb_rel, ub_rel):

        # init_lb: (1, 28, 28) or (3, 32, 32)
        
        init_lb = init_lb.flatten()
        init_ub = init_ub.flatten()

        out_shape = lb_rel.shape[:-1]

        lb_rel = torch.flatten(lb_rel, start_dim=0, end_dim=-2)
        ub_rel = torch.flatten(ub_rel, start_dim=0, end_dim=-2)

        lb_res = torch.empty(lb_rel.shape[0])
        ub_res = torch.empty(ub_rel.shape[0])

        # given input matrix and input bounds compute output bounds
        for i in range(ub_rel.shape[0]):

                ub_temp = ub_rel[i,:-1].clone()
                lb_temp = lb_rel[i,:-1].clone()
                ub_b = ub_rel[i,-1].clone()
                lb_b = lb_rel[i,-1].clone()

                init_ub_temp = init_ub.clone()
                init_lb_temp = init_lb.clone()

                init_ub_temp[ub_temp < 0] = init_lb[ub_temp < 0]
                init_lb_temp[ub_temp < 0] = init_ub[ub_temp < 0]

                ub_res[i] = ub_temp @ init_ub_temp + ub_b

                init_ub_temp = init_ub.clone()
                init_lb_temp = init_lb.clone()

                init_ub_temp[lb_temp < 0] = init_lb[lb_temp < 0]
                init_lb_temp[lb_temp < 0] = init_ub[lb_temp < 0]

                lb_res[i] = lb_temp @ init_lb_temp + lb_b

        #view thign not necessary, would hide an error if it was there
        
        #lb_res = lb_res.view(out_shape)
        #ub_res = ub_res.view(out_shape)

        return lb_res, ub_res

def check_postcondition(init_lb, init_ub, lb_rel, ub_rel, true_label):
        init_lb = torch.flatten(init_lb)
        init_ub = torch.flatten(init_ub)

        differences = torch.empty((0, lb_rel.shape[-1]))
        for i in range(10):
                if i == true_label:
                        continue
                curr_diff = lb_rel[true_label] - ub_rel[i]
                curr_diff = curr_diff.unsqueeze(0)
                differences = torch.cat((differences, curr_diff), dim=0)

        assert differences.shape[0] == 9

        # lower bounds of differences must be positive
        numerical_diff = torch.empty((0,))
        for i in range(9):
                lb_temp = init_lb.clone()
                row = differences[i]
                bias = row[-1]
                row = row[:-1]

                lb_temp[row < 0] = init_ub[row < 0]

                diff_num = torch.sum(row * lb_temp) + bias

                numerical_diff = torch.cat((numerical_diff, diff_num.unsqueeze(0)), dim=0)

        #print(numerical_diff)
        
        return int(numerical_diff.min() >= 0)