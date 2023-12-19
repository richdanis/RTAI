import argparse
import torch
import torch.nn as nn

import time

from networks import get_network
from utils.loading import parse_spec
from transforms import *

DEVICE = "cpu"


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

    #create a list with a 1 in the ith position if the ith position is a relu layer
    relu_list = []

    #create a list wiht the input sizes for the conv_layers. should be a list of tuples
    conv_input_shapes = []
    conv_input_shapes.append((1,28,28))

    for layer in net:
        if isinstance(layer, nn.ReLU):
            relu_list.append(1)
        else:
            relu_list.append(0)
    #delete the first element of relu_list, because the first layer isnt counted as a layer
    relu_list.pop(0)
    #add a 0 at the end of relu_list, because the last layer isnt counted as a relu layer
    relu_list.append(0)

    #create lists for the relational upper and lower bounds. the ith element is the tensor of relational bounds in layer i
    lbounds = []
    ubounds = []
    rel_lbounds = []
    rel_ubounds = []

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
        return backsubstitute_upper(layer_nr - 1, res_lb, res_ub)


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

            start_time = time.time()

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
            
            end_time= time.time()
            
            start_time_2= time.time()
            
            init_lb_temp = init_lb.clone()
            init_ub_temp = init_ub.clone()
            coeffs_new= res_lb[:,:-1]   
            bias_fast= res_lb[:,-1]
            
            coeffs_bool_low = coeffs_new<0 
            coeffs_bool_up  = coeffs_new>0
                       
            coeffs_bool_low= coeffs_bool_low.int()    
            coeffs_bool_up = coeffs_bool_up.int()
            
            init_lb_temp = torch.unsqueeze(init_lb_temp, dim=0)
            init_ub_temp = torch.unsqueeze(init_ub_temp, dim=0)

# Use expand or repeat to replicate the vector along the new dimension
            init_lb_temp = init_lb_temp.expand(res_lb.shape[0], -1)
            init_ub_temp = init_ub_temp.expand(res_lb.shape[0], -1)
            
            low= torch.multiply(init_ub_temp, coeffs_bool_low)
            up = torch.multiply(init_lb_temp, coeffs_bool_up)
            
            matrix = low+up #100x784
            
            #selected_rows_coeff = torch.index_select(coeffs_new, dim=0)
            res= torch.einsum("ij,ji->i",matrix, coeffs_new.T)
           
            #res = torch.matmul(matrix, selected_rows_coeff) #should have shape 100x1 but 100x784 @ 100x784 -> 100x1
            res = res+bias_fast   #100x1
            end_time_2= time.time()
            
            if torch.allclose(res,rel_lb):
                print("success")
                diff= end_time-start_time
                diff_2= end_time_2- start_time_2
                print(f'duration old approach {diff}, duration with matrix operation {diff_2}')
                print(f'increase {diff/diff_2} times')
            else:
                print("fail")
            
            assert False

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
        

        #Seems to work for layer_nr = 0 -> Box bounds are the same as backsubstituted relational bounds
        # if layer_nr == 0:
        #     rel_lb = torch.zeros(res_lb.shape[0])
        #     rel_ub = torch.zeros(res_ub.shape[0])


        #     for i in range(res_lb.shape[0]):
        #         init_lb_temp = init_lb.clone()
        #         init_ub_temp = init_ub.clone()
        #         coeffs = res_lb[i, :-1]
        #         bias = res_lb[i, -1]

        #         # #TODO Exchange j-th entry of init_ub with jth entry of init_lb if  jth entry of coeffs is negative
        #         # for j in range(coeffs.shape[0]):
        #         #     if coeffs[j] < 0:
        #         #         init_ub_temp[j], init_lb_temp[j] = init_lb_temp[j].clone(), init_ub_temp[j].clone()
                    
        #         negative_coeffs_mask = coeffs < 0
        #         init_ub_temp[negative_coeffs_mask], init_lb_temp[negative_coeffs_mask] = init_lb_temp[negative_coeffs_mask].clone(), init_ub_temp[negative_coeffs_mask].clone()
        #         #compute the i-th entry of the tensor res_ub and res_lb
                
        #         rel_lb[i] = torch.matmul(init_lb_temp, coeffs)
        #         rel_lb[i] += bias
        #         #rel_ub[i] = torch.matmul(init_ub_temp, coeffs) + bias


        #     for i in range(res_ub.shape[0]):
        #         init_lb_temp = init_lb.clone()
        #         init_ub_temp = init_ub.clone()
        #         coeffs = res_ub[i, :-1]
        #         bias = res_ub[i, -1]

        #         # #TODO Exchange j-th entry of init_ub with jth entry of init_lb if  jth entry of coeffs is negative
        #         # for j in range(coeffs.shape[0]):
        #         #     if coeffs[j] < 0:
        #         #         init_ub_temp[j], init_lb_temp[j] = init_lb_temp[j].clone(), init_ub_temp[j].clone()
        #         negative_coeffs_mask = coeffs < 0
        #         init_ub_temp[negative_coeffs_mask], init_lb_temp[negative_coeffs_mask] = init_lb_temp[negative_coeffs_mask].clone(), init_ub_temp[negative_coeffs_mask].clone()
                    
        #         #compute the i-th entry of the tensor res_ub and res_lb
        #         rel_ub[i] = torch.matmul(init_ub_temp, coeffs) + bias

        #     assert (rel_lb.shape == rel_ub.shape)
        #     assert torch.all(rel_lb <= rel_ub)
        
        #     return rel_lb, rel_ub
        if layer_nr == 0:
            init_lb_temp = init_lb.clone()
            init_ub_temp = init_ub.clone()

            coeffs = res_lb[:, :-1]
            bias = res_lb[:, -1]

            # Step 1: Create a 2D tensor `init` where each row is a copy of `init_ub_temp`
            init = init_lb_temp.unsqueeze(0).expand_as(coeffs)

            # Step 2: Create a mask where `coeffs` is less than 0
            negative_coeffs_mask = coeffs < 0

            # Step 3: Use the mask to replace the corresponding elements in `init` with the corresponding elements from `init_lb_temp`
            init[negative_coeffs_mask] = init_ub_temp.unsqueeze(0).expand_as(coeffs)[negative_coeffs_mask]

            # Now `init` is a 2D tensor where each row i is composed of elements from `init_ub_temp` and `init_lb_temp` according to the rule: 
            # If `coeffs[i][j]` > 0 then `init[i][j]` is `init_ub_temp[j]`, otherwise `init[i][j]` is `init_lb_temp[j]`.

            res_lb_1 = torch.mul(init, coeffs).sum(dim=1) 
            res_lb_1 = torch.add(res_lb_1, bias)

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
        return backsubstitute_upper(layer_nr - 1, res_lb, res_ub)





    # ub_rel = torch.eye(init_lb.numel())
    # ub_rel = torch.reshape(ub_rel, (init_lb.shape[0], init_lb.shape[1], init_lb.shape[2], init_lb.shape[0] * init_lb.shape[1] * init_lb.shape[2]))
    # add constant 0 for bias
    # shape = ub_rel.shape[:-1] + (1,)
    # ub_rel = torch.cat((ub_rel, torch.zeros(shape)), dim=-1)
    # lb_rel = ub_rel.detach().clone()


    # propagate box through network

    for layer in net:
        layer_nr +=1

        #debugging: Assert that the elements rel_ubounds and rel_lbounds dont change after ther have been assigned
        #assert (len(rel_ubounds) == layer_nr)
        if isinstance(layer, nn.Flatten):
            ub = torch.flatten(ub)
            lb = torch.flatten(lb)
            layer_nr -=1
        elif isinstance(layer, nn.Linear):
            time1 = time.time()
            with torch.no_grad():
                if(layer_nr == 0):
                    lb, ub = transform_linear(lb, ub, layer.weight, layer.bias)

                else:
                    lb, ub = transform_linear(lb_1, ub_1, layer.weight, layer.bias)

                lb_rel, ub_rel = transform_linear_rel( layer.weight, layer.bias)
            
                lb_1, ub_1 = backsubstitute_upper_nonsparse(layer_nr, lb_rel, ub_rel)

                assert torch.all(lb <= ub)
                #assert torch.all(lb_1 -lb >=-0.01)
                #cvc bc xassert torch.all(ub -ub_1 >=-0.01)


            assert(torch.all(lb_1 <= ub_1))


            #assert(max(lb -lb_1) < 1e-3)
            #assert(max(ub_1 -ub) < 1e-3)
            lbounds.append(lb_1)
            ubounds.append(ub_1)

            rel_lbounds.append(lb_rel)
            rel_ubounds.append(ub_rel)
            assert torch.all(lb <= ub)
            time2 = time.time()
            print("time for linear layer: ", layer_nr, time2 - time1)
        elif isinstance(layer, nn.Conv2d):  
            time1 = time.time()

            lb_rel, ub_rel, weight, bias, out_shape = transform_conv_rel(layer, input_shape = conv_input_shapes[-1][1:])

            ub = torch.flatten(ub)
            lb = torch.flatten(lb)

            lb, ub = transform_conv(lb, ub, weight, bias)
            assert torch.all(lb <= ub)

            lb_1, ub_1 = backsubstitute_upper_nonsparse(layer_nr, lb_rel, ub_rel)
            assert(torch.all(lb_1 <= ub_1))
            assert torch.all(lb_1 -lb >=-0.01)
            assert torch.all(ub -ub_1 >=-0.01)

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
            time2 = time.time()
            print("time for conv layer: ", layer_nr, time2 - time1)

        elif isinstance(layer, nn.ReLU):
            lb, ub, lb_rel, ub_rel = transform_relu_rel(lb_1, ub_1, lb_rel, ub_rel)
            lb_1, ub_1 = backsubstitute_upper_nonsparse(layer_nr, lb_rel, ub_rel)


            assert torch.all(lb <= ub)
            assert lb_rel.shape == ub_rel.shape
            lbounds.append(lb_1)
            ubounds.append(ub_1)
            rel_lbounds.append(lb_rel)
            rel_ubounds.append(ub_rel)

        elif isinstance(layer, nn.LeakyReLU):
            lb, ub, lb_rel, ub_rel = transform_leaky_relu_rel(lb_1, ub_1, lb_rel, ub_rel, negslope = layer.negative_slope)
            lb_1, ub_1 = backsubstitute_upper_nonsparse(layer_nr, lb_rel, ub_rel)


            assert torch.all(lb <= ub)
            assert lb_rel.shape == ub_rel.shape
            lbounds.append(lb_1)
            ubounds.append(ub_1)
            rel_lbounds.append(lb_rel)
            rel_ubounds.append(ub_rel)
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
        
    #debugging: How much did backsubtitiotion work? ---> Lb and ub seem to be very very close to each other


    #lb, ub = backsubstitute_upper(layer_nr, rel_lbounds[layer_nr], rel_ubounds[layer_nr])
        

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
    with torch.no_grad():
        lb, ub = transform_linear(lb_1, ub_1, weight, bias)
        lb_rel, ub_rel = transform_linear_rel( weight, bias)
        lb_1, ub_1 = backsubstitute_upper_nonsparse(layer_nr +1, lb_rel, ub_rel)

    #2) check if the lower bound of the target output is greater than

    ver = torch.all(lb_1 >= 0)
    ver = bool(ver)
    if ver:
        print("differece between lb and max ub of other classes: ", torch.sort(lb_1)[0][1])
        return(bool(ver))
    else:
        print("differece between lb and max ub of other classes: ", lb_1.min())
        return(bool(ver))

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
