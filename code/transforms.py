import torch
import torch.nn as nn

def transform_linear(lb_rel, ub_rel, weight, bias):
        """
        Propagate relational bounds through a linear layer.
        """
        neg_weights = torch.where(weight < 0, weight, torch.zeros_like(weight))
        pos_weights = torch.where(weight > 0, weight, torch.zeros_like(weight))
                                  
        lb_res = pos_weights @ lb_rel + neg_weights @ ub_rel
        ub_res = pos_weights @ ub_rel + neg_weights @ lb_rel

        lb_res[:, -1] = lb_res[:, -1] + bias
        ub_res[:, -1] = ub_res[:, -1] + bias

        return lb_res, ub_res

def matrix_matrix_mul_rel(lb_rel, ub_rel, weight, bias):
    """
    Propagate an abstract box through a matrix multiplication.
    """
    # Separate positive and negative parts of the weight
    weight_pos = torch.where(weight > 0, weight, torch.zeros_like(weight))
    weight_neg = torch.where(weight < 0, weight, torch.zeros_like(weight))

    # Compute lower bound result
    lb_res = torch.einsum('ij,jak->iak', weight_neg, ub_rel) + torch.einsum('ij,jak->iak', weight_pos, lb_rel)
    lb_res[..., -1] += bias.unsqueeze(1)

    # Compute upper bound result
    ub_res = torch.einsum('ij,jak->iak', weight_neg, lb_rel) + torch.einsum('ij,jak->iak', weight_pos, ub_rel)
    ub_res[..., -1] += bias.unsqueeze(1)

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
        ind = ind.long()

        lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)
        ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)

        # flatten input
        lb_unfold = lb_rel[ind]
        ub_unfold = ub_rel[ind]

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
        in_shape = lb_rel.shape
        in_shape_lb = lb.shape
        #check that lb and lb_rel have same batch size
        assert(in_shape[0]== in_shape_lb[0])
        #asssert that lb is smaller than ub
        assert(torch.all(lb <= ub))

        upper_slope = ub / (ub - lb)
        offset = upper_slope * lb
        offset = offset.flatten()

        ub_res = upper_slope.unsqueeze(-1) * ub_rel

        # flatten ub, ub_rel
        ub = ub.flatten(start_dim=0)
        ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)
        ub_res = ub_res.flatten(start_dim=0, end_dim=-2)

        # flatten lb, lb_rel
        lb = lb.flatten(start_dim=0)
        lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)
        lb_res = lb_rel.flatten(start_dim=0, end_dim=-2)

        ub_res[:,-1] = ub_res[:,-1] - offset

        lb_res[:,:] = 0

        lb_res[ub < 0,:] = 0
        ub_res[ub < 0,:] = 0

        ub_res[lb > 0,:] = ub_rel[lb > 0,:]
        lb_res[lb > 0,:] = lb_rel[lb > 0,:]

        lb_res = lb_res.view(in_shape)
        ub_res = ub_res.view(in_shape)

        return lb_res, ub_res

def transform_ReLU_alpha(lb_rel, ub_rel, lb, ub, alpha):

        in_shape = lb_rel.shape
        in_shape_lb = lb.shape
        #check that lb and lb_rel have same batch size
        assert(in_shape[0]== in_shape_lb[0])

        upper_slope = ub / (ub - lb)

        # offset!!!
        offset = upper_slope * lb
        offset = offset.flatten()
        ub_res = upper_slope.unsqueeze(-1) * ub_rel

        # flatten ub, ub_rel
        ub = ub.flatten(start_dim=0)
        ub_rel = ub_rel.flatten(start_dim=0, end_dim=-2)
        ub_res = ub_res.flatten(start_dim=0, end_dim=-2)

        # flatten lb, lb_rel
        lb = lb.flatten(start_dim=0)
        lb_rel = lb_rel.flatten(start_dim=0, end_dim=-2)
        lb_res = lb_rel.flatten(start_dim=0, end_dim=-2)

        ub_res[:,-1] = ub_res[:,-1] - offset
        
        lb_res = alpha.flatten().unsqueeze(1) * lb_rel

        lb_res[ub < 0,:] = 0
        ub_res[ub < 0,:] = 0

        ub_res[lb > 0,:] = ub_rel[lb > 0,:]
        lb_res[lb > 0,:] = lb_rel[lb > 0,:]

        lb_res = lb_res.view(in_shape)
        ub_res = ub_res.view(in_shape)

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
        
        init_lb = init_lb.view((-1,))
        init_ub = init_ub.view((-1,))

        out_shape = lb_rel.shape[:-1]

        lb_rel = lb_rel.view((-1, lb_rel.shape[-1]))
        ub_rel = ub_rel.view((-1, ub_rel.shape[-1]))

        neg_lb = torch.where(lb_rel < 0, lb_rel, torch.zeros_like(lb_rel))
        pos_lb = torch.where(lb_rel > 0, lb_rel, torch.zeros_like(lb_rel))

        neg_ub = torch.where(ub_rel < 0, ub_rel, torch.zeros_like(ub_rel))
        pos_ub = torch.where(ub_rel > 0, ub_rel, torch.zeros_like(ub_rel))

        lb_res = neg_lb[:,:-1] @ init_ub + pos_lb[:,:-1] @ init_lb
        ub_res = neg_ub[:,:-1] @ init_lb + pos_ub[:,:-1] @ init_ub

        lb_res = lb_res + lb_rel[:,-1]
        ub_res = ub_res + ub_rel[:,-1]

        return lb_res.view(out_shape), ub_res.view(out_shape)

def differences(init_lb, init_ub, lb_rel, ub_rel, true_label):
        
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
        neg_diff = torch.where(differences < 0, differences, torch.zeros_like(differences))
        pos_diff = torch.where(differences > 0, differences, torch.zeros_like(differences))

        numerical_diff = neg_diff[:,:-1] @ init_ub + pos_diff[:,:-1] @ init_lb
        numerical_diff = numerical_diff + differences[:,-1]
 
        return numerical_diff