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
        
def propagate_linear_symbolic(inputs, fc: nn.Linear):
        """
        Propagate an abstract box through a linear layer.
        """
        # inputs is matrix of form (input_neurons, n_symbols)
        # multiply each column of weight matrix with each row of inputs and sum
        # then add bias
        # output is matrix of form (output_neurons, n_symbols)
        # TODO: make this faster
        res = torch.zeros(fc.weight.shape[0], inputs.shape[1])
        for i in range(fc.weight.shape[0]):
                # multiply each column of weight matrix with each row of inputs
               temp = fc.weight[i,:].unsqueeze(1) * inputs
               # sum over rows
               temp = temp.sum(dim=0)
               temp[-1] = temp[-1] + fc.bias[i]
               res[i] = temp
        return res

def propagate_linear_fast(inputs, fc: nn.Linear):
       # this implementation is slower than propagate_linear_symbolic
       inputs = inputs.unsqueeze(0)
       inputs = inputs.repeat(fc.weight.shape[0], 1, 1)
       res = fc.weight.unsqueeze(2) * inputs
       res = res.sum(dim=1)
       res[:,-1] = res[:,-1] + fc.bias
       return res
