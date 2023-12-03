import torch
import torch.nn as nn


def propagate_linear_rel(lb_rel, ub_rel, weight, bias):
        """
        Propagate relational bounds through a linear layer.
        """
        lb_res = torch.empty(weight.shape[0], lb_rel.shape[1])
        ub_res = torch.empty(weight.shape[0], lb_rel.shape[1])
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

def propagate_linear_symbolic(inputs, weight, bias):
        """
        Propagate a relational equation through a linear layer.
        """
        # inputs is matrix of form (input_neurons, n_symbols)
        # multiply each column of weight matrix with each row of inputs and sum
        # then add bia
        res = torch.zeros(weight.shape[0], inputs.shape[1])
        for i in range(weight.shape[0]):
                # multiply each column of weight matrix with each row of inputs
               temp = weight[i,:].unsqueeze(1) * inputs
               # sum over rows
               temp = temp.sum(dim=0)
               temp[-1] = temp[-1] + bias[i]
               res[i] = temp
        return res

def propagate_linear_symbolic_fast(inputs, weight, bias):
       # this implementation is slower than propagate_linear_symbolic
       inputs = inputs.unsqueeze(0)
       inputs = inputs.repeat(weight.shape[0], 1, 1)
       res = weight.unsqueeze(2) * inputs
       res = res.sum(dim=1)
       res[:,-1] = res[:,-1] + bias
       return res

def matrix_matrix_mul_symbolic(inputs, weight, bias):
        """
        Propagate an abstract box through a matrix multiplication.
        """
        res = torch.zeros(weight.shape[0], inputs.shape[1], inputs.shape[2])
        for i in range(weight.shape[0]):
                weight_row = weight[i,:].unsqueeze(1)
                for j in range(inputs.shape[1]):
                        # multiply each column of weight matrix with each row of inputs
                        temp =  weight_row* inputs[:,j,:]
                        # sum over rows
                        temp = temp.sum(dim=0)
                        temp[-1] = temp[-1] + bias[i]
                        res[i,j] = temp
        return res

def propagate_conv2d_symbolic(inputs, conv: nn.Conv2d):
        """
        Propagate an abstract box through a convolutional layer.
        Input shape: (in_channels, height, width, number_of_symbols)
        """

        assert len(inputs.shape) == 4
        # get number of channels
        out_channels = conv.weight.shape[0]
        kernel_size = conv.weight.shape[2]
        stride = conv.stride[0]
        padding = conv.padding[0]

        # compute output shape
        out_height = (inputs.shape[1] + 2 * padding - kernel_size) // stride + 1
        out_width = (inputs.shape[2] + 2 * padding - kernel_size) // stride + 1
        out_shape = (out_channels, out_height, out_width, inputs.shape[-1])

        # index array
        shape = torch.tensor(inputs.shape)
        num_ind = shape[:-1].prod()
        ind = torch.arange(0, num_ind, dtype=torch.float)
        ind = ind.reshape(inputs.shape[:-1])

        # unfold index array
        ind = torch.nn.functional.unfold(ind, (kernel_size, kernel_size), stride=stride, padding=padding)
        # change to int
        ind = ind.int()

        # flatten input
        inputs = inputs.flatten(start_dim=0, end_dim=-2)

        assert len(ind.shape) == 2
        # ind is now of shape (in_channels * kernel_size * kernel_size, num_patches)
        # unfold input

        unfolded = torch.empty(ind.shape + (inputs.shape[-1],))
        for i in range(ind.shape[0]):
                for j in range(ind.shape[1]):
                        unfolded[i, j] = inputs[ind[i, j]]

        # get weight and bias
        w = conv.weight
        w = w.view(w.shape[0], -1)
        # w is now of shape (out_channels, in_channels * kernel_size * kernel_size)
        b = conv.bias
        # b is of shape (out_channels,)

        # pass weight, bias and unfolded input through linear layer
        # issue here is that we have a matrix matrix multiplication and not a matrix vector multiplication
        res = matrix_matrix_mul_symbolic(unfolded, w, b)
        assert len(res.shape) == 3
        # reshape to output shape
        res = res.view(out_shape)

        return res

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

def propagate_conv2d_rel(lb_rel, ub_rel, conv: nn.Conv2d):
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

def propagate_ReLU_rel(lb_rel, ub_rel, lb, ub):
        upper_slope = ub / (ub - lb)

        ub_rel = upper_slope.unsqueeze(-1) * (ub_rel - lb.unsqueeze(-1))

        # flatten ub, ub_rel
        ub = ub.flatten(start_dim=0)
        ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)

        ub_rel[ub < 0,:] = 0

        # flatten lb, lb_rel
        lb = lb.flatten(start_dim=0)
        lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)

        lb_rel[lb < 0,:] = 0

        lb_rel = lb_rel.view(ub_rel.shape)
        ub_rel = ub_rel.view(ub_rel.shape)

        return lb_rel, ub_rel

def propagate_ReLU_rel_alpha(lb_rel, ub_rel, lb, ub, alpha):

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

def propagate_leakyReLU_rel(lb_rel, ub_rel, lb, ub, slope, alpha = 1):
        
        
        lb_rel_bef = lb_rel.clone()
        ub_rel_bef = ub_rel.clone()
        s = slope
        
        if (slope <=1):
                upper_slope = (ub - s*lb) / (ub - lb)
                #lower_slope = torch.ones_like(upper_slope)

                ub = ub.unsqueeze_(-1)
                lb = lb.unsqueeze_(-1)
                upper_slope = upper_slope.unsqueeze_(-1)
                #lower_slope = lower_slope.unsqueeze_(-1)

                ub_rel = upper_slope * ub_rel + lb*(s - upper_slope)

                lb_rel = alpha * lb_rel.clone()

                # flatten ub, ub_rel
                ub = ub.flatten(start_dim=0)
                ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)

                ub_rel[ub < 0,:] = s*ub_rel_bef[ub < 0,:].clone()
                #lb_rel[ub < 0,:] = s*lb_rel_bef[ub < 0,:].clone()

                # flatten lb, lb_rel
                lb = lb.flatten(start_dim=0)
                lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)

                #lb_rel_bef = lb_rel_bef.flatten(start_dim=0, end_dim=-2)
                lb_rel[lb > 0,:] = lb_rel_bef[lb > 0,:].clone()
                #ub_rel[lb > 0,:] = ub_rel_bef[lb > 0,:].clone()

                lb_rel = lb_rel.view(lb_rel.shape)
                ub_rel = ub_rel.view(ub_rel.shape)

                return lb_rel, ub_rel

        elif (slope> 1):
                upper_slope = (ub - lb)/ub
                lower_slope = (ub - lb*s)/(ub - lb)
                ub = ub.unsqueeze_(-1)
                lb = lb.unsqueeze_(-1)
                upper_slope = upper_slope.unsqueeze_(-1)
                lower_slope = lower_slope.unsqueeze_(-1)

                ub_rel = upper_slope * ub_rel - lb*upper_slope

                lb_rel = lower_slope * lb_rel + lb*(s - lower_slope)

                # flatten ub, ub_rel
                ub = ub.flatten(start_dim=0)
                ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)

                ub_rel[ub < 0,:] = s* ub_rel_bef[ub < 0,:].clone()
                lb_rel[ub < 0,:] = s* lb_rel_bef[ub < 0,:].clone()


                # flatten lb, lb_rel
                lb = lb.flatten(start_dim=0)
                lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)

                #lb_rel_bef = lb_rel_bef.flatten(start_dim=0, end_dim=-2)
                lb_rel[lb > 0,:] = lb_rel_bef[lb > 0,:].clone()
                ub_rel[lb > 0,:] = ub_rel_bef[lb > 0,:].clone()

                lb_rel = lb_rel.view(lb_rel.shape)
                ub_rel = ub_rel.view(ub_rel.shape)

                return lb_rel, ub_rel


def propagate_leakyReLU_rel(lb_rel, ub_rel, lb, ub, slope, alpha = 1):
        lb_rel_bef = lb_rel.clone()
        ub_rel_bef = ub_rel.clone()
        lb_rel_bef = lb_rel_bef.flatten(start_dim=0, end_dim=-2)
        ub_rel_bef = ub_rel_bef.flatten(start_dim=0, end_dim=-2)
        s = slope



        
        if (slope <=1.0):
                upper_slope = (ub - s*lb) / (ub - lb)
                ub = ub.unsqueeze_(-1)
                lb = lb.unsqueeze_(-1)
                upper_slope = upper_slope.unsqueeze_(-1)
                #lower_slope = lower_slope.unsqueeze_(-1)

                #lower_slope = torch.ones_like(upper_slope)


                #lower_slope = lower_slope.unsqueeze_(-1)

                ub_rel = upper_slope * ub_rel + lb*(s - upper_slope)

                lb_rel = alpha * lb_rel.clone()

                

                # flatten ub, ub_rel
                ub = ub.flatten(start_dim=0)
                ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)

                ub_rel[ub < 0,:] = s*ub_rel_bef[ub < 0,:].clone()
                #lb_rel[ub < 0,:] = s*lb_rel_bef[ub < 0,:].clone()

                # flatten lb, lb_rel
                lb = lb.flatten(start_dim=0)
                lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)

                #lb_rel_bef = lb_rel_bef.flatten(start_dim=0, end_dim=-2)
                lb_rel[lb > 0,:] = lb_rel_bef[lb > 0,:].clone()
                #ub_rel[lb > 0,:] = ub_rel_bef[lb > 0,:].clone()

                lb_rel = lb_rel.view(lb_rel.shape)
                ub_rel = ub_rel.view(ub_rel.shape)

                return lb_rel, ub_rel

        elif (slope> 1.0):
                upper_slope = (ub - lb)/ub
                lower_slope = (ub - lb*s)/(ub - lb)
                ub = ub.unsqueeze_(-1)
                lb = lb.unsqueeze_(-1)
                upper_slope = upper_slope.unsqueeze_(-1)
                lower_slope = lower_slope.unsqueeze_(-1)

                ub_rel = upper_slope * ub_rel - lb*upper_slope

                lb_rel = lower_slope * lb_rel + lb*(s - lower_slope)

                # flatten ub, ub_rel
                ub = ub.flatten(start_dim=0)
                ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)

                ub_rel[ub < 0,:] = s* ub_rel_bef[ub < 0,:].clone()
                lb_rel[ub < 0,:] = s* lb_rel_bef[ub < 0,:].clone()


                # flatten lb, lb_rel
                lb = lb.flatten(start_dim=0)
                lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)

                #lb_rel_bef = lb_rel_bef.flatten(start_dim=0, end_dim=-2)
                lb_rel[lb > 0,:] = lb_rel_bef[lb > 0,:].clone()
                ub_rel[lb > 0,:] = ub_rel_bef[lb > 0,:].clone()

                lb_rel = lb_rel.view(lb_rel.shape)
                ub_rel = ub_rel.view(ub_rel.shape)

                return lb_rel, ub_rel



        

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

        lb_res = lb_res.view(out_shape)
        ub_res = ub_res.view(out_shape)

        return lb_res, ub_res