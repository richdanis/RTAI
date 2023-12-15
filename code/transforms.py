import torch
import torch.nn as nn

def transform_linear(lb, ub, weight, bias):
        """
        Propagate non-relational bounds through a linear layer.
        lb and ub are of shape [784] in case of MNIST

        """

        lb = lb.repeat(weight.shape[0], 1)
        ub = ub.repeat(weight.shape[0], 1)
        assert lb.shape == ub.shape == weight.shape


        mul_lb = torch.where(weight > 0, lb, ub)
        mul_ub = torch.where(weight > 0, ub, lb)

        lb = (mul_lb * weight).sum(dim=1)
        ub = (mul_ub * weight).sum(dim=1)
        assert lb.shape == ub.shape == bias.shape

        if bias is not None:
            lb += bias
            ub += bias

        #lb = lb.unsqueeze(0)
        #ub = ub.unsqueeze(0)

        return lb, ub



def transform_linear_rel(weight, bias):
        """
        Propagate relational bounds through a linear layer.
        should return the weight matrix and the bias vector appended as the last column of the weight matrix

        """
        lb_shape = [weight.shape[0], weight.shape[1] + 1]
        ub_shape = [weight.shape[0], weight.shape[1]+1]
        

        lb_rel = torch.cat((weight, bias.unsqueeze(-1)), dim=-1)
        ub_rel = torch.cat((weight, bias.unsqueeze(-1)), dim=-1)

        # assert(lb_rel.shape == ub_rel.shape)
        # assert(lb_rel.shape == lb_shape)
        # assert(ub_rel.shape == ub_shape)

        return lb_rel, ub_rel

def transform_relu_rel(lb, ub, lb_rel, ub_rel, lambd = 0.0):
        """ 
        
        Propagate (non- and relational) bounds through a ReLU layer.
        shape of lb and ub should be preserved. 
        lb_rel and ub_rel are the relational bounds of the inputs of the relu

        lambda is now a vector

        """
        #Lambda population option 1: the ith element of lmbd should be zero is the ith element of ub is bigger than -the ith element of lb, otherwise 1
        lambd = torch.where(ub <= -lb, torch.zeros_like(ub), torch.ones_like(ub))

        res_lb = torch.empty(lb.shape)
        res_ub = torch.empty(ub.shape)
        res_lb_rel = torch.empty(lb.shape[0], lb.shape[0] + 1)
        res_ub_rel = torch.empty(ub.shape[0], ub.shape[0] + 1)

        for i in range(lb.shape[0]):
                if lb[i] >= 0:
                        res_lb[i] = lb[i]
                        res_ub[i] = ub[i]
                        #two alternatives: 1. set lb_rel and ub_rel to a zeros vetor to a one in the i-th position or 2. keep them as the same relational bound
                        #option 2 is correct! The rel_bounds are wrt previous neurons, not previous previous neurons
                        res_lb_rel[i, :] = torch.zeros_like(res_lb_rel[i, :])
                        res_lb_rel[i, i] = 1
                        res_ub_rel[i, :] = torch.zeros_like(res_ub_rel[i, :])
                        res_ub_rel[i, i] = 1
                elif ub[i] <= 0:
                        res_lb[i] = 0
                        res_ub[i] = 0
                        res_lb_rel[i, :] = torch.zeros_like(res_lb_rel[i, :])
                        res_ub_rel[i, :] = torch.zeros_like(res_ub_rel[i, :])
                else:
                        res_lb[i] = lambd[i] * lb[i]
                        res_ub[i] = ub[i]
                        #same two alternatives here
                        res_lb_rel[i, :] = torch.zeros_like(res_lb_rel[i, :])
                        res_lb_rel[i, i] = lambd[i]

                        res_ub_rel[i, :] = torch.zeros_like(res_ub_rel[i, :])
                        res_ub_rel[i, i] = ub[i]/(ub[i] - lb[i])
                        res_ub_rel[i, -1] = -(lb[i]*ub[i])/(ub[i] - lb[i])
                

        #assert(res_lb_rel.shape == lb_rel.shape)
        #assert(res_ub_rel.shape == ub_rel.shape)
        #assert(res_lb.shape == res_ub.shape == res_lb_rel.shape == res_ub_rel.shape)



        return res_lb, res_ub, res_lb_rel, res_ub_rel

def transform_leaky_relu_rel(lb, ub, lb_rel, ub_rel, negslope,  lambd = 0.0):
        """ 
        
        Propagate (non- and relational) bounds through a ReLU layer.
        shape of lb and ub should be preserved. 
        lb_rel and ub_rel are the relational bounds of the inputs of the relu

        lambda is now a vector

        """

        if (negslope <= 1.0):
                #TODO(1): Figure out a decission rule for the lambda values --> May not be the same as for normal relu
                lambd = torch.where(ub <= -lb, negslope*torch.ones_like(ub), torch.ones_like(ub))

                #for lambda learning: lambd has to be between negslope and 1 (in the case negslope <=1)

                res_lb = torch.empty(lb.shape)
                res_ub = torch.empty(ub.shape)
                res_lb_rel = torch.empty(lb.shape[0], lb.shape[0] + 1)
                res_ub_rel = torch.empty(ub.shape[0], ub.shape[0] + 1)

                for i in range(lb.shape[0]):
                        if lb[i] >= 0:
                                res_lb[i] = lb[i]
                                res_ub[i] = ub[i]
                                #two alternatives: 1. set lb_rel and ub_rel to a zeros vetor to a one in the i-th position or 2. keep them as the same relational bound
                                #option 2 is correct! The rel_bounds are wrt previous neurons, not previous previous neurons
                                res_lb_rel[i, :] = torch.zeros_like(res_lb_rel[i, :])
                                res_lb_rel[i, i] = 1
                                res_ub_rel[i, :] = torch.zeros_like(res_ub_rel[i, :])
                                res_ub_rel[i, i] = 1
                        elif ub[i] <= 0:
                                res_lb[i] = negslope*lb[i]
                                res_ub[i] = negslope*ub[i]

                                res_lb_rel[i, :] = torch.zeros_like(res_lb_rel[i, :])
                                res_lb_rel[i, i] = negslope
                                res_ub_rel[i, :] = torch.zeros_like(res_ub_rel[i, :])
                                res_ub_rel[i, i] = negslope
                        else:
                                res_lb[i] = lambd[i] * lb[i] #### need to check it later
                                res_ub[i] = ub[i]
                                #same two alternatives here
                                res_lb_rel[i, :] = torch.zeros_like(res_lb_rel[i, :])
                                res_lb_rel[i, i] = lambd[i]

                                res_ub_rel[i, :] = torch.zeros_like(res_ub_rel[i, :])
                                res_ub_rel[i, i] = 1 - (lb[i]*(negslope - 1))/(ub[i] - lb[i])
                                res_ub_rel[i, -1] = (lb[i]*(negslope - 1))/(1-(lb[i]/ub[i]))
                        

                #assert(res_lb_rel.shape == lb_rel.shape)
                #assert(res_ub_rel.shape == ub_rel.shape)
                #assert(res_lb.shape == res_ub.shape == res_lb_rel.shape == res_ub_rel.shape)

                return res_lb, res_ub, res_lb_rel, res_ub_rel
        
        elif (negslope > 1.0):
                 #TODO(1): Figure out a decission rule for the lambda values --> May not be the same as for normal relu
                lambd = torch.where(ub <= -lb, negslope*torch.ones_like(ub), torch.ones_like(ub))

                #for lambda learning: lambd has to be between negslope and 1 (in the case negslope <=1)

                res_lb = torch.empty(lb.shape)
                res_ub = torch.empty(ub.shape)
                res_lb_rel = torch.empty(lb.shape[0], lb.shape[0] + 1)
                res_ub_rel = torch.empty(ub.shape[0], ub.shape[0] + 1)

                for i in range(lb.shape[0]):
                        if lb[i] >= 0:
                                res_lb[i] = lb[i]
                                res_ub[i] = ub[i]
                                #two alternatives: 1. set lb_rel and ub_rel to a zeros vetor to a one in the i-th position or 2. keep them as the same relational bound
                                #option 2 is correct! The rel_bounds are wrt previous neurons, not previous previous neurons
                                res_lb_rel[i, :] = torch.zeros_like(res_lb_rel[i, :])
                                res_lb_rel[i, i] = 1
                                res_ub_rel[i, :] = torch.zeros_like(res_ub_rel[i, :])
                                res_ub_rel[i, i] = 1
                        elif ub[i] <= 0:
                                res_lb[i] = negslope*lb[i]
                                res_ub[i] = negslope*ub[i]

                                res_lb_rel[i, :] = torch.zeros_like(res_lb_rel[i, :])
                                res_lb_rel[i, i] = negslope
                                res_ub_rel[i, :] = torch.zeros_like(res_ub_rel[i, :])
                                res_ub_rel[i, i] = negslope
                        else:
                                res_lb[i] = negslope * lb[i] #### need to check it later
                                res_ub[i] = lambd[i] * ub[i]
                                #same two alternatives here

                                res_ub_rel[i, :] = torch.zeros_like(res_lb_rel[i, :])
                                res_ub_rel[i, i] = lambd[i]

                                res_lb_rel[i, :] = torch.zeros_like(res_ub_rel[i, :])
                                res_lb_rel[i, i] = 1 - (lb[i]*(negslope - 1))/(ub[i] - lb[i])
                                res_lb_rel[i, -1] = (lb[i]*(negslope - 1))/(1-(lb[i]/ub[i]))


                return res_lb, res_ub, res_lb_rel, res_ub_rel



        else:
                raise NotImplementedError("The negative slope should be a positive number")
  



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

        #set the etnries of upper_slope that are negative to zero
        upper_slope = upper_slope.clamp_min(0)


        ub_rel = upper_slope.unsqueeze(-1) * (ub_rel - lb.unsqueeze(-1))

        lb_rel = torch.zeros_like(lb_rel)



        # flatten ub, ub_rel --> Dont need to flatten it!
        #ub = ub.flatten(start_dim=0)
        #ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)

        ub_rel[ub <= 0,:] = torch.zeros_like(ub_rel[ub <= 0,:])
        lb_rel[ub <= 0,:] = torch.zeros_like(lb_rel[ub <= 0,:])

        # flatten lb, lb_rel
        #lb = lb.flatten(start_dim=0)
        #lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)

        lb_rel[lb >= 0,:] = lb_rel_before[lb >= 0,:]    
        ub_rel[lb >= 0,:] = ub_rel_before[lb >= 0,:]

        #lb_rel = lb_rel.view(in_shape)
        #ub_rel = ub_rel.view(in_shape)

        return lb_rel, ub_rel

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

        #view thign not necessary, would hide an error if it was there

        #lb_res = lb_res.view(out_shape)
        #ub_res = ub_res.view(out_shape)

        return lb_res, ub_res