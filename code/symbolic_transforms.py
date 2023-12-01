import torch
import torch.nn as nn

        
def propagate_linear_symbolic(inputs, weight, bias):
        """
        Propagate an abstract box through a linear layer.
        """
        # inputs is matrix of form (input_neurons, n_symbols)
        # multiply each column of weight matrix with each row of inputs and sum
        # then add bias
        # output is matrix of form (output_neurons, n_symbols)
        # print("inputs:",inputs.shape)
        # print("weight matrix:", weight.shape)
        # print("bias:", bias.shape)
        # assert False
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

def check_postcondition(lb, ub, true_label):
        """
        Check if the post-condition is satisfied.
        """
        ub[:,true_label] = -float("inf")
        return lb[:,true_label] > ub.max()