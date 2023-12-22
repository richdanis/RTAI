import argparse
import torch
import torch.nn as nn

import time

from networks import get_network
from utils.loading import parse_spec
from transforms_alpha import *
from torch.optim import SGD

DEVICE = "cpu"
torch.set_default_dtype(torch.float64)

def output_diff_lb2ub(lb, ub, tue_label):
    "return difference between lower bound of target and max upper bound of other classes"
    ub[tue_label] = -float("inf")
    return float(lb[tue_label] - ub.max())


def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    
    # input shape:
    # mnist (1, 28, 28) -> x_1 , ..., x_784
    # cifar10 (3, 32, 32) -> 1_1, ... , x_3072

    # SETUP

    # set weights to no_grad:
    for param in net.parameters():
        param.requires_grad = False

    #set input to no_grad:
    inputs.requires_grad = False

    #define init_lb and init_ub tensors with no_grad:

    lb = torch.clamp(inputs - eps, 0, 1)
    ub = torch.clamp(inputs + eps, 0, 1)
    init_lb = lb.clone()
    init_ub = ub.clone()
    init_lb = init_lb.flatten()
    init_ub = init_ub.flatten()
    init_lb.requires_grad = False
    init_ub.requires_grad = False
    lb.requires_grad = False
    ub.requires_grad = False

    layer_nr = -1 #dont count flatten as layer, use python nr: 0-th layer is 1st linear layer


    #create a list wiht the input sizes for the conv_layers. should be a list of tuples
    conv_input_shapes = []
    
    conv_input_shapes.append(tuple(inputs.shape))


    #create lists for the relational upper and lower bounds. the ith element is the tensor of relational bounds in layer i
    lbounds = []
    ubounds = []
    rel_lbounds = []
    rel_ubounds = []


    #create list where the alpha values are stored -> ith element correcponds to alpha vector in the ith layer

    first_pass = True
    relu_nr = 0
    leak_relu_nr = 0

    alphas = []
    alphas_leak = []
    neg_slopes = []
    
    #alphas = [torch.zeros(0, requires_grad = True)] * (len(net) -1)
    #alphas_1 = torch.ones(len(net) -1, 0)

    #define loss fucntion
    def loss_fn(lb_1, alphas):
        
        return  - sum(lb_1)
    
    #define optimizer
    #optimizer = torch.optim.Adam(alphas, lr = 0.01)


    def backsubstitute_upper_orig(layer_nr, res_lb, res_ub):
        """
        assume current layer is layer_nr = i+1, has m Neurons and previous layer is layer_nr = i, has n Neurons and layer i-1 has k Neurons
        -> ub_rel[i+1].shape = (m, n+1)
        -> ub_rel[i].shape = (n, k+1)

        create a tensor res that has shape (m, k+1)
        every of the m rows looks like this (assumnig all entries in ub_rel[i+1][j, :-1] are non-negative]):
        ub_rel[i+1][j, :-1] @ ub_rel[i][:,1] , ub_rel[i+1][j, :-1] @ ub_rel[i][:,2] , ...,  ub_rel[i+1][j, :-1] @  ub_rel[i][:,-1] + ub_rel[i+1][j, -1]

        if the lth entry in ub_rel[i+1][j, :-1] is negative, the lth row of ub_rel[i] hast to be exchanged with the lth row of lb_rel[i]
        

        """
        #Seems to work for layer_nr = 0 -> Box bounds are the same as backsubstituted relational bounds
        if layer_nr == 0:
            rel_lb = torch.zeros(res_lb.shape[0])
            rel_ub = torch.zeros(res_ub.shape[0])


            for i in range(res_lb.shape[0]):
                init_lb_temp = init_lb.clone()
                init_ub_temp = init_ub.clone()
                coeffs = res_lb[i, :-1]
                bias = res_lb[i, -1]

                #TODO Exchange j-th entry of init_ub with jth entry of init_lb if  jth entry of coeffs is negative
                for j in range(coeffs.shape[0]):
                    if coeffs[j] < 0:
                        init_ub_temp[j], init_lb_temp[j] = init_lb_temp[j].clone(), init_ub_temp[j].clone()
                    
                #compute the i-th entry of the tensor res_ub and res_lb
                rel_lb[i] = torch.matmul(init_lb_temp, coeffs)
                rel_lb[i] += bias
                #rel_ub[i] = torch.matmul(init_ub_temp, coeffs) + bias


            for i in range(res_ub.shape[0]):
                init_lb_temp = init_lb.clone()
                init_ub_temp = init_ub.clone()
                coeffs = res_ub[i, :-1]
                bias = res_ub[i, -1]

                #TODO Exchange j-th entry of init_ub with jth entry of init_lb if  jth entry of coeffs is negative
                for j in range(coeffs.shape[0]):
                    if coeffs[j] < 0:
                        init_ub_temp[j], init_lb_temp[j] = init_lb_temp[j].clone(), init_ub_temp[j].clone()
                    
                #compute the i-th entry of the tensor res_ub and res_lb
                rel_ub[i] = torch.matmul(init_ub_temp, coeffs) + bias

            assert (rel_lb.shape == rel_ub.shape)
            assert torch.all(rel_lb <= rel_ub)
        
            return rel_lb, rel_ub

                        
        #backsubstitute upper bounds from layer layer_nr to layer 0
        rel_ub = res_ub.clone()
        rel_lb = res_lb.clone()

        rel_ub_prev = rel_ubounds[layer_nr - 1].clone()
        rel_lb_prev = rel_lbounds[layer_nr - 1].clone()
        rel_ub_prev_1 = rel_ubounds[layer_nr - 1].clone()
        rel_lb_prev_1 = rel_lbounds[layer_nr - 1].clone()

        res_ub = torch.zeros((rel_ub.shape[0], rel_ub_prev.shape[1]))
        res_lb = torch.zeros((rel_lb.shape[0], rel_lb_prev.shape[1]))

        #pupulate res_lb
        for i in range(rel_lb.shape[0]):
            
            # if relu_list[layer_nr] == 1:
            #     coeffs = rel_ub[i, :]
            # else:
            #     coeffs = rel_lb[i, :-1]
            #     bias = rel_lb[i, -1]
            coeffs = rel_lb[i, :-1]
            bias = rel_lb[i, -1]

            rel_ub_prev_temp = rel_ub_prev.clone()
            rel_lb_prev_temp = rel_lb_prev.clone()
            

            #if the jth entry of coeffs is negative, exchange the jth row of rel_ub_prev with the jth row of rel_lb_prev
            #TODO: check if this is correct with relus: here bounds are allways the same
            for j in range(coeffs.shape[0]):
                if coeffs[j] < 0:
                    rel_ub_prev_temp[j, :], rel_lb_prev_temp[j, :] = rel_lb_prev_temp[j, :].clone(), rel_ub_prev_temp[j,:].clone()

            #compute the tensor res_ub that has shape (m, k+1)
            for j in range(rel_ub_prev.shape[1]):
                res_lb[i][j] = torch.matmul(rel_lb_prev_temp[:, j], coeffs)

            # if relu_list[layer_nr] == 0:
            #     res_lb[i][-1] = torch.matmul(rel_lb_prev_temp[:,-1], coeffs) + bias
            res_lb[i][-1] = torch.matmul(rel_lb_prev_temp[:,-1], coeffs) + bias



        #pupulate res_ub
        for i in range(rel_ub.shape[0]):

            # if relu_list[layer_nr] == 1:
            #     coeffs = rel_ub[i, :]
            # else:
            #     coeffs = rel_lb[i, :-1]
            #     bias = rel_lb[i, -1]

            coeffs = rel_lb[i, :-1]
            bias = rel_lb[i, -1]

            rel_ub_prev_temp = rel_ub_prev.clone()
            rel_lb_prev_temp = rel_lb_prev.clone()

            #if the jth entry of coeffs is negative, exchange the jth row of rel_ub_prev with the jth row of rel_lb_prev
            #TODO: check if this is correct with relus: here bounds are allways the same
            for j in range(coeffs.shape[0]):
                if coeffs[j] < 0:
                    rel_ub_prev_temp[j, :], rel_lb_prev_temp[j, :] = rel_lb_prev_temp[j, :].clone(), rel_ub_prev_temp[j, :].clone()

            #compute the tensor res_ub that has shape (m, k+1)
            for j in range(rel_lb_prev.shape[1]):
                res_ub[i][j] = torch.matmul(rel_ub_prev_temp[:, j], coeffs)
            
            # if relu_list[layer_nr] == 0:
            #     res_ub[i][-1] = torch.matmul(rel_ub_prev_temp[:,-1], coeffs) + bias

            res_ub[i][-1] = torch.matmul(rel_ub_prev_temp[:,-1], coeffs) + bias
        

        assert (res_lb.shape == res_ub.shape)
        #assert torch.all(res_lb == res_ub)
        return backsubstitute_upper_orig(layer_nr - 1, res_lb, res_ub)


    def backsubstitute_upper_nonsparse(layer_nr, res_lb, res_ub):
        """
        assume current layer is layer_nr = i+1, has m Neurons and previous layer is layer_nr = i, has n Neurons and layer i-1 has k Neurons
        -> ub_rel[i+1].shape = (m, n+1)
        -> ub_rel[i].shape = (n, k+1)

        create a tensor res that has shape (m, k+1)
        every of the m rows looks like this (assumnig all entries in ub_rel[i+1][j, :-1] are non-negative]):
        ub_rel[i+1][j, :-1] @ ub_rel[i][:,1] , ub_rel[i+1][j, :-1] @ ub_rel[i][:,2] , ...,  ub_rel[i+1][j, :-1] @  ub_rel[i][:,-1] + ub_rel[i+1][j, -1]

        if the lth entry in ub_rel[i+1][j, :-1] is negative, the lth row of ub_rel[i] hast to be exchanged with the lth row of lb_rel[i]
        

        """
        

        #Seems to work for layer_nr = 0 -> Box bounds are the same as backsubstituted relational bounds
        if layer_nr == 0:
            rel_lb = torch.zeros(res_lb.shape[0])
            rel_ub = torch.zeros(res_ub.shape[0])


            for i in range(res_lb.shape[0]):
                init_lb_temp = init_lb.clone()
                init_ub_temp = init_ub.clone()
                coeffs = res_lb[i, :-1]
                bias = res_lb[i, -1]

                # #TODO Exchange j-th entry of init_ub with jth entry of init_lb if  jth entry of coeffs is negative
                # for j in range(coeffs.shape[0]):
                #     if coeffs[j] < 0:
                #         init_ub_temp[j], init_lb_temp[j] = init_lb_temp[j].clone(), init_ub_temp[j].clone()
                    
                negative_coeffs_mask = coeffs < 0
                init_ub_temp[negative_coeffs_mask], init_lb_temp[negative_coeffs_mask] = init_lb_temp[negative_coeffs_mask].clone(), init_ub_temp[negative_coeffs_mask].clone()
                #compute the i-th entry of the tensor res_ub and res_lb
                
                rel_lb[i] = torch.matmul(init_lb_temp, coeffs)
                rel_lb[i] += bias
                #rel_ub[i] = torch.matmul(init_ub_temp, coeffs) + bias

            # #Chris stuff 
            # coeffs_lb_bool= coeffs <0 
            # coeff_upp_bool= not coeffs_lb_bool
            
            # coeffs_lb_bool= int(coeffs_lb_bool)
            # coeff_upp_bool= int(coeff_upp_bool)
            # # True -> 1 , False ->0
            
            # res= torch.multiply(init_lb_temp,coeffs_lb_bool)+ torch.multiply(init_ub_temp,coeff_upp_bool)
            # res = torch.sum(res, dim=0)
            # res= res+bias
            
            # print(res, "res matrix operation")
            # print(res_lb, "for loop")
            
            # if res== res_lb:
            #     print("success")
            
            # assert False


            for i in range(res_ub.shape[0]):
                init_lb_temp = init_lb.clone()
                init_ub_temp = init_ub.clone()
                coeffs = res_ub[i, :-1]
                bias = res_ub[i, -1]

                # #TODO Exchange j-th entry of init_ub with jth entry of init_lb if  jth entry of coeffs is negative
                # for j in range(coeffs.shape[0]):
                #     if coeffs[j] < 0:
                #         init_ub_temp[j], init_lb_temp[j] = init_lb_temp[j].clone(), init_ub_temp[j].clone()
                negative_coeffs_mask = coeffs < 0
                init_ub_temp[negative_coeffs_mask], init_lb_temp[negative_coeffs_mask] = init_lb_temp[negative_coeffs_mask].clone(), init_ub_temp[negative_coeffs_mask].clone()
                    
                #compute the i-th entry of the tensor res_ub and res_lb
                rel_ub[i] = torch.matmul(init_ub_temp, coeffs) + bias

            assert (rel_lb.shape == rel_ub.shape)
            assert torch.all(rel_lb <= rel_ub)
        
            return rel_lb, rel_ub

                        
        #backsubstitute upper bounds from layer layer_nr to layer 0
        rel_ub = res_ub.clone()
        rel_lb = res_lb.clone()

        rel_ub_prev = rel_ubounds[layer_nr - 1].clone()
        rel_lb_prev = rel_lbounds[layer_nr - 1].clone()
        # rel_ub_prev_1 = rel_ubounds[layer_nr - 1].clone()
        # rel_lb_prev_1 = rel_lbounds[layer_nr - 1].clone()

        res_ub = torch.zeros((rel_ub.shape[0], rel_ub_prev.shape[1]))
        res_lb = torch.zeros((rel_lb.shape[0], rel_lb_prev.shape[1]))



        #pupulate res_lb
        for i in range(rel_lb.shape[0]):
            
            # if relu_list[layer_nr] == 1:
            #     coeffs = rel_ub[i, :]
            # else:
            #     coeffs = rel_lb[i, :-1]
            #     bias = rel_lb[i, -1]
            coeffs = rel_lb[i, :-1]
            bias = rel_lb[i, -1]

            rel_ub_prev_temp = rel_ub_prev.clone()
            rel_lb_prev_temp = rel_lb_prev.clone()
            

            #if the jth entry of coeffs is negative, exchange the jth row of rel_ub_prev with the jth row of rel_lb_prev
            #TODO: check if this is correct with relus: here bounds are allways the same
            negative_coeffs_mask = coeffs < 0
            rel_ub_prev_temp[negative_coeffs_mask], rel_lb_prev_temp[negative_coeffs_mask] = rel_lb_prev[negative_coeffs_mask], rel_ub_prev[negative_coeffs_mask]
                  

            #compute the tensor res_ub that has shape (m, k+1)
            # for j in range(rel_ub_prev.shape[1]):
            #     res_lb[i][j] = torch.matmul(rel_lb_prev_temp[:, j], coeffs)
            res_lb[i] = torch.matmul(rel_lb_prev_temp.t(), coeffs.unsqueeze(-1)).squeeze(-1)


            # if relu_list[layer_nr] == 0:
            #     res_lb[i][-1] = torch.matmul(rel_lb_prev_temp[:,-1], coeffs) + bias
            res_lb[i][-1] = torch.matmul(rel_lb_prev_temp[:,-1], coeffs)
            res_lb[i][-1].add_(bias)



        #pupulate res_ub
        for i in range(rel_ub.shape[0]):

            # if relu_list[layer_nr] == 1:
            #     coeffs = rel_ub[i, :]
            # else:
            #     coeffs = rel_lb[i, :-1]
            #     bias = rel_lb[i, -1]

            coeffs = rel_lb[i, :-1]
            bias = rel_lb[i, -1]

            rel_ub_prev_temp = rel_ub_prev.clone()
            rel_lb_prev_temp = rel_lb_prev.clone()

            #if the jth entry of coeffs is negative, exchange the jth row of rel_ub_prev with the jth row of rel_lb_prev
            #TODO: check if this is correct with relus: here bounds are allways the same
            negative_coeffs_mask = coeffs < 0
            rel_ub_prev_temp[negative_coeffs_mask], rel_lb_prev_temp[negative_coeffs_mask] = rel_lb_prev[negative_coeffs_mask], rel_ub_prev[negative_coeffs_mask]
                  

            #compute the tensor res_ub that has shape (m, k+1)
            # for j in range(rel_lb_prev.shape[1]):
            #     res_ub[i][j] = torch.matmul(rel_ub_prev_temp[:, j], coeffs)

            res_ub[i] = torch.matmul(rel_ub_prev_temp.t(), coeffs.unsqueeze(-1)).squeeze(-1)
            
            # if relu_list[layer_nr] == 0:
            #     res_ub[i][-1] = torch.matmul(rel_ub_prev_temp[:,-1], coeffs) + bias

            res_ub[i][-1] = torch.matmul(rel_ub_prev_temp[:,-1], coeffs)
            res_ub[i][-1].add_(bias)
        

        assert (res_lb.shape == res_ub.shape)
        #assert torch.all(res_lb == res_ub)
        return backsubstitute_upper_nonsparse(layer_nr - 1, res_lb, res_ub)
    
    def backsubstitute_upper(layer_nr, res_lb, res_ub):
        """
        assume current layer is layer_nr = i+1, has m Neurons and previous layer is layer_nr = i, has n Neurons and layer i-1 has k Neurons
        -> ub_rel[i+1].shape = (m, n+1)
        -> ub_rel[i].shape = (n, k+1)

        create a tensor res that has shape (m, k+1)
        every of the m rows looks like this (assumnig all entries in ub_rel[i+1][j, :-1] are non-negative]):
        ub_rel[i+1][j, :-1] @ ub_rel[i][:,1] , ub_rel[i+1][j, :-1] @ ub_rel[i][:,2] , ...,  ub_rel[i+1][j, :-1] @  ub_rel[i][:,-1] + ub_rel[i+1][j, -1]

        if the lth entry in ub_rel[i+1][j, :-1] is negative, the lth row of ub_rel[i] hast to be exchanged with the lth row of lb_rel[i]
        

        """
        
        

        if layer_nr == 0:
            init_lb_temp = init_lb.clone()
            init_ub_temp = init_ub.clone()

            coeffs = res_lb[:, :-1]
            bias = res_lb[:, -1]

            lb_mul = torch.where(coeffs <= 0, init_ub_temp, init_lb_temp)

            rel_lb = (lb_mul * coeffs).sum(dim = 1)
            rel_lb += bias

            coeffs = res_ub[:, :-1]
            bias = res_ub[:, -1]
            
            rb_mul = torch.where(coeffs <= 0, init_lb_temp, init_ub_temp)
            rel_ub = (rb_mul * coeffs).sum(dim = 1)
            rel_ub += bias



            assert (rel_lb.shape == rel_ub.shape)
            #assert torch.all(rel_lb <= rel_ub)

            return rel_lb, rel_ub

        # rel_ub = res_ub.clone()
        # rel_lb = res_lb.clone()  
        # rel_ub_prev = rel_ubounds[layer_nr - 1].clone()
        # rel_lb_prev = rel_lbounds[layer_nr - 1].clone()
        # # rel_ub_prev_1 = rel_ubounds[layer_nr - 1].clone()
        # # rel_lb_prev_1 = rel_lbounds[layer_nr - 1].clone()

        # res_ub = torch.zeros((rel_ub.shape[0], rel_ub_prev.shape[1]))
        # res_lb = torch.zeros((rel_lb.shape[0], rel_lb_prev.shape[1]))        

        # #pupulate res_lb
        # coeffs = rel_lb[:, :-1]
        # bias = rel_lb[:, -1]

        # #ith entry of res_lb is sum of 1st column of rel_lb_prev * ith column of coeffs, 2nd column of rel_lb_prev * ith column of coeffs, ..., last column of rel_lb_prev * ith row of coeffs

        # sum(coeffs *rel_lb_prev)
 

        #############################


        rel_ub = res_ub.clone()
        rel_lb = res_lb.clone()

        rel_ub_prev = rel_ubounds[layer_nr - 1].clone()
        rel_lb_prev = rel_lbounds[layer_nr - 1].clone()
        # rel_ub_prev_1 = rel_ubounds[layer_nr - 1].clone()
        # rel_lb_prev_1 = rel_lbounds[layer_nr - 1].clone()

        res_ub = torch.zeros((rel_ub.shape[0], rel_ub_prev.shape[1]))
        res_lb = torch.zeros((rel_lb.shape[0], rel_lb_prev.shape[1]))

        #pupulate res_lb

        #create the matrix weight
        coeffs_lower = rel_lb[:, :-1]
        coeffs_lower_pos = torch.where(coeffs_lower <0, torch.zeros_like(coeffs_lower), coeffs_lower) 
        coeffs_lower_inv = torch.where(coeffs_lower <0, coeffs_lower, torch.zeros_like(coeffs_lower))
        bias = rel_lb[:, -1]


        


        weights = torch.cat((coeffs_lower_pos, coeffs_lower_inv), dim = 1)

        #create the matrix weight_prev (take bias out)!
        weight_prev = torch.cat((rel_lb_prev[:,:-1], rel_ub_prev[:,:-1]), dim = 0)
        bias_prev = torch.cat((rel_lb_prev[:,-1], rel_ub_prev[:,-1]), dim = 0)

        #compute the matrix multiplication
        res_lb[:,:-1] = weights @ weight_prev

        #compute the bias
        res_lb[:, -1] = weights @ bias_prev
        res_lb[:, -1] += bias


        assert res_lb.shape == (rel_ub.shape[0], rel_ub_prev.shape[1])


        #pupulate res_ub

        #create the matrix weight
        coeffs_upper = rel_ub[:, :-1]
        coeffs_upper_pos = torch.where(coeffs_upper <0, torch.zeros_like(coeffs_upper), coeffs_upper)
        coeffs_upper_inv = torch.where(coeffs_upper <0, coeffs_upper, torch.zeros_like(coeffs_upper))
        bias = rel_ub[:, -1]

        weights = torch.cat((coeffs_upper_pos, coeffs_upper_inv), dim = 1)

        #create the matrix weight_prev (take bias out)!
        weight_prev = torch.cat((rel_ub_prev[:,:-1], rel_lb_prev[:,:-1]), dim = 0)
        bias_prev = torch.cat((rel_ub_prev[:,-1], rel_lb_prev[:,-1]), dim = 0)

        #compute the matrix multiplication
        res_ub[:,:-1] = weights @ weight_prev

        #compute the bias
        res_ub[:, -1] = weights @ bias_prev
        res_ub[:, -1] += bias

        assert res_ub.shape == (rel_ub.shape[0], rel_ub_prev.shape[1])
        
        

        assert (res_lb.shape == res_ub.shape)
        #assert torch.all(res_lb == res_ub)
        return backsubstitute_upper(layer_nr - 1, res_lb, res_ub)

 
    # propagate box through network

    for i in range(20):
        layer_nr = -1

        for layer in net:
            layer_nr +=1

            #debugging: Assert that the elements rel_ubounds and rel_lbounds dont change after ther have been assigned
            #assert (len(rel_ubounds) == layer_nr)
            if isinstance(layer, nn.Flatten):
                if layer_nr == 0:
                    lb = init_lb.clone()
                    ub = init_ub.clone()

                ub = torch.flatten(ub)
                lb = torch.flatten(lb)
                layer_nr -=1
            elif isinstance(layer, nn.Linear):
                time1 = time.time()
                
                if(layer_nr == 0):
                    lb, ub = transform_linear(lb, ub, layer.weight, layer.bias)

                else:
                    lb, ub = transform_linear(lb_1, ub_1, layer.weight, layer.bias)

                lb_rel, ub_rel = transform_linear_rel( layer.weight, layer.bias)
            
                lb_1, ub_1 = backsubstitute_upper(layer_nr, lb_rel, ub_rel)

                assert torch.all(lb <= ub)
                #assert torch.all(lb_1 -lb >=-0.01)
                #assert torch.all(ub -ub_1 >=-0.01)


                #assert(torch.all(lb_1 <= ub_1))


                #assert(max(lb -lb_1) < 1e-3)
                #assert(max(ub_1 -ub) < 1e-3)
                #lbounds.append(lb_1)
                #ubounds.append(ub_1)

                rel_lbounds.append(lb_rel)
                rel_ubounds.append(ub_rel)
                #assert torch.all(lb <= ub)
                #print("time for linear layer: ", layer_nr, time2 - time1)
            elif isinstance(layer, nn.Conv2d):  

                lb_rel, ub_rel, weight, bias, out_shape = transform_conv_rel(layer, input_shape = conv_input_shapes[-1])



                if layer_nr == 0:
                    lb_1, ub_1 = lb, ub

                ub = torch.flatten(ub_1)
                lb = torch.flatten(lb_1)

                lb, ub = transform_conv(lb, ub, weight, bias)

                assert torch.all(lb <= ub)
                lb_1, ub_1 = backsubstitute_upper(layer_nr, lb_rel, ub_rel)

                assert(torch.all(lb_1 <= ub_1))
                #assert torch.all(lb_1 -lb >=-0.01)
                #assert torch.all(ub -ub_1 >=-0.01)

                lbounds.append(lb_1)
                ubounds.append(ub_1)
                rel_lbounds.append(lb_rel)
                rel_ubounds.append(ub_rel)

                #calculate the input shape for the next layer

                conv_input_shapes.append(out_shape)

                # height, width = conv_input_shapes[-1]
                # output_height = (height - layer.kernel_size[0] + 2*layer.padding[0]) // layer.stride[0] + 1
                # output_width = (width - layer.kernel_size[0] + 2*layer.padding[0]) // layer.stride[0] + 1

                # conv_input_shapes.append([output_height, output_width])


            elif isinstance(layer, nn.ReLU):
                if first_pass == True:
                    lambd = torch.where(ub_1 <= -lb_1, torch.zeros_like(ub), torch.ones_like(ub))
                    lambd.requires_grad = True
                    alphas.append(lambd)
                    alphas[relu_nr].requires_grad = True
                    alphas[relu_nr].retain_grad()

                lb, ub, lb_rel, ub_rel = transform_relu_rel(lb_1, ub_1, lb_rel, ub_rel, lambd = alphas[relu_nr])
                lb_1, ub_1 = backsubstitute_upper(layer_nr, lb_rel, ub_rel)


                assert torch.all(lb <= ub)
                assert lb_rel.shape == ub_rel.shape
                # assert torch.all(lb_1 -lb >=-0.01)
                # assert torch.all(ub -ub_1 >=-0.01)
                lbounds.append(lb_1)
                ubounds.append(ub_1)
                rel_lbounds.append(lb_rel)
                rel_ubounds.append(ub_rel)
                relu_nr +=1

            elif isinstance(layer, nn.LeakyReLU):
                if first_pass == True:
                    lambd = torch.where(ub_1 <= -lb_1, layer.negative_slope*torch.ones_like(ub), torch.ones_like(ub))

                    lambd.requires_grad = True
                    alphas_leak.append(lambd)
                    alphas_leak[leak_relu_nr].requires_grad = True
                    alphas_leak[leak_relu_nr].retain_grad()
                if layer.negative_slope <= 1:
                    assert torch.all(alphas_leak[leak_relu_nr] >= layer.negative_slope)
                    assert torch.all(alphas_leak[leak_relu_nr] <= 1)

                if layer.negative_slope > 1:
                    assert torch.all(alphas_leak[leak_relu_nr] <= layer.negative_slope)
                    assert torch.all(alphas_leak[leak_relu_nr] >= 1)



                lb, ub, lb_rel, ub_rel = transform_leaky_relu_rel(lb_1, ub_1, lb_rel, ub_rel, negslope = layer.negative_slope, lambd = alphas_leak[leak_relu_nr])
                lb_1, ub_1 = backsubstitute_upper(layer_nr, lb_rel, ub_rel)
                neg_slopes.append(layer.negative_slope)


                #assert torch.all(lb <= ub)
                assert lb_rel.shape == ub_rel.shape
                lbounds.append(lb_1)
                ubounds.append(ub_1)
                rel_lbounds.append(lb_rel)
                rel_ubounds.append(ub_rel)
                leak_relu_nr +=1
            else:
                raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
            
        relu_nr = 0
        leak_relu_nr = 0
        
        
        #chck verification new way 

        #1) Treat it as a linear layer where the i-th output is the different between true_label output 
        #and the i-th output of the network
        #weight matrix should have minus ones at the diagonal and the column of tru_labe should be ones

        #bias is zero
        weight =  -1* torch.eye(10)
        ones = torch.ones(10, 10)
        j = true_label
        weight[:, j] = ones[:, j]
        weight[j,j] = 0
        

        bias = torch.zeros(10)
        
        lb, ub = transform_linear(lb_1, ub_1, weight, bias)
        lb_rel, ub_rel = transform_linear_rel( weight, bias)
        lb_1, ub_1 = backsubstitute_upper(len(net) -1, lb_rel, ub_rel)
        rel_lbounds = []
        rel_ubounds = []

        #2) check if the lower bound of the target output is greater than

        ver = torch.all(lb_1 >= 0)
        ver = bool(ver)
        if ver:
            print("differece between lb and max ub of other classes: ", torch.sort(lb_1)[0][1])
            return(bool(ver))
        else:
            print("differece between lb and max ub of other classes: ", lb_1.min())

            if first_pass == True and len(alphas) > 0:
                optimizer_relu = torch.optim.Adam(alphas, lr = 0.2)
            if first_pass == True and len(alphas_leak) > 0:
                optimizer_leaky = torch.optim.Adam(alphas_leak, lr = 0.2)
            first_pass = False

            if len(alphas) > 0:

                optimizer_relu.zero_grad()
                loss = loss_fn(lb_1, alphas)
                loss.retain_grad()
                loss.backward(retain_graph = True)
                optimizer_relu.step()
                with torch.no_grad():
                    for i in range(len(alphas)):
                        alphas[i] = alphas[i].clamp_( min=0, max=1)
                        alphas[i].retain_grad()
                print("1 step learning")
                print("current loss" , loss)

            if len(alphas_leak) > 0:
                optimizer_leaky.zero_grad()
                loss = loss_fn(lb_1, alphas_leak)
                loss.retain_grad()
                loss.backward(retain_graph = True)
                optimizer_leaky.step()
                with torch.no_grad():
                    for i in range(len(alphas_leak)):
                        if neg_slopes[i] <=1:
                            alphas_leak[i] = alphas_leak[i].clamp_( min=neg_slopes[i], max=1)
                            alphas_leak[i].retain_grad()
                        elif neg_slopes[i] > 1:
                            alphas_leak[i] = alphas_leak[i].clamp_( min=1, max=neg_slopes[i])
                            alphas_leak[i].retain_grad()
                print("1 step learning")
                print("current loss" , loss)

            
            

            
        
            
        
        
    #debugging: How much did backsubtitiotion work? ---> Lb and ub seem to be very very close to each other


    #lb, ub = backsubstitute_upper(layer_nr, rel_lbounds[layer_nr], rel_ubounds[layer_nr])
        

    #chck verification new way 

    #1) Treat it as a linear layer where the i-th output is the different between true_label output 
    #and the i-th output of the network
    #weight matrix should have minus ones at the diagonal and the column of tru_labe should be ones

    # #bias is zero
    # weight =  -1* torch.eye(10)
    # ones = torch.ones(10, 10)
    # j = true_label
    # weight[:, j] = ones[:, j]
    # weight[j,j] = 0
    

    # bias = torch.zeros(10)
    # with torch.no_grad():
    #     lb, ub = transform_linear(lb_1, ub_1, weight, bias)
    #     lb_rel, ub_rel = transform_linear_rel( weight, bias)
    #     lb_1, ub_1 = backsubstitute_upper(len(net) -1, lb_rel, ub_rel)

    # #2) check if the lower bound of the target output is greater than

    # ver = torch.all(lb_1 >= 0)
    # ver = bool(ver)
    # if ver:
    #     print("differece between lb and max ub of other classes: ", torch.sort(lb_1)[0][1])
    #     return(bool(ver))
    # else:
    #     print("differece between lb and max ub of other classes: ", lb_1.min())
    #     return(bool(ver))

    #return bool(ver)


    #checck verification old school
    # lb_true = lb_1[true_label]--net fc_6 --spec test_cases/fc_6/img0_mnist_0.0495.txt
    # ub_1[true_label] = -float("inf")
    # print("differece between lb and max ub of other classes: ", float(lb_true - ub_1.max()))
    # return (lb_true > ulb_1b_1.max())



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

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    #check how much time it takes to run the analysze function

    start = time.time()

    try:
        if analyze(net, image, eps, true_label):
            print("verified")
        else:
            print("not verified")
    except Exception as e:
        print("Exception occurred during analysis: ", e)

    end = time.time()
    print("time: ", end - start)


if __name__ == "__main__":
    main()
