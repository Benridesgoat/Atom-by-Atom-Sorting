#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:02:01 2022

@author: robert
"""

from routeoptimizerbrowaeys import *
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import bvn

from Other.Classes.TrapSiteClass import *
from Other.Classes.AtomClass import *
from Other.Classes.LaserClass import *

def create_square_template(dim_in,low):
    """
    Creates a square template of dim_in x dim_in spots with the left bottom
    corner at [low,low]

    Parameters
    ----------
    dim_in : int
        dimension of the grid
    low : int
        left corner located at [low,low]

    Returns
    -------
    template: np.array of ints with dimension dim_in**2 x 2 
        array with all the locations of the template

    """
    template=np.zeros([dim_in**2,2])
    for k in range(dim_in):
        for l in range(dim_in):
            template[dim_in*k+l,:]=[int(low+k),int(low+l)]
    template=np.int64(template)
    return template

def create_square_grid(dim_out):
    """
    creates a grid of possible spots as dim_out x dim_out

    Parameters
    ----------
    dim_out : int
        dimension of the square grid

    Returns
    -------
    grid: np.array of ints with dimension dim_out**2 x 2 
        array with all the locations of the grid

    """
    grid=np.zeros([dim_out**2,2])
    for k in range(dim_out):
        for l in range(dim_out):
            grid[dim_out*k+l,:]=[k,l]
    grid=np.int64(grid)
    return grid

#np.random.seed(10)

for dim_in in range(5,6):     #10,11
    #dim_in = 11     #makes it run faster
    dim_out = int(np.ceil(2**(1/2)*dim_in))+1    #+2 for lp simulations, +1 for time pickup-move sims
    num_atoms = 0 #np.int64(dim_out**2/2)
    
    R=1
    
    #Trapping laser description
    grid=create_square_grid(dim_out)
    template=create_square_template(dim_in,(dim_out-dim_in)/2)
    num_traps=len(grid)
    TrapLasers=np.ndarray([num_traps],dtype=object)
    Traps=np.ndarray([num_traps],dtype=object)
    
    laser_efficiency=0.10
    laser_decrease_factor=0.02
    P=np.zeros([num_traps])+8
    P0=P*laser_efficiency*laser_decrease_factor
    w0=np.zeros([num_traps])+0.8*10**(-6) #waist-size of laser
    wavelength=np.zeros([num_traps])+813.035*10**(-9) #wave length trap in nm
    n_ref=1
    polarization=0
    z_traps=0
    sim_traps=True
    Env_Temp=0
    
    #Atom state space description
    spin_basis=np.array([[5,0,0,0,0],[5,1,0,0,1],[61,0,1,0,1]],np.int32)
    spin_basis_labels=["0","1","r"]
    motional_basis=[[0,0,0]]
    motional_basis_labels=["000"]
    Temp=0
    interacting_states=[2]
    dims=[[len(spin_basis),len(motional_basis)]*num_atoms]*2
    
    
    Atoms=np.ndarray([num_atoms],dtype=object)
    sim_traps=True
    
    for k in range(num_traps):
        TrapLasers[k]=Laser(P0[k],w0[k],polarization,wavelength[k])
        Traps[k]=Trap_Site(grid[k][0], grid[k][1], z_traps, TrapLasers[k], n_ref)
    
    
    it_create_assignment = 1
    #it_seed = 6
    #lambda_adjusted = 1.015
    #testls = np.zeros([10,40])
    move_pickup_array = [[50,600]]
    for move_pickup in range(len(move_pickup_array)):
        plot_data = np.zeros([20,90,2])  #ammount of i's, ammount of lambda's, 2. 20-90-2
        #count=0
        for i in range(0,1):    #0-20
            for lambda_adjusted100 in range(256,257):   #10-100 for time puckup-move sims
                lambda_adjusted=lambda_adjusted100/100
        #for regexp in range(1,25): #1 tot 25 is fine  #tab all below if needed
            #reg=10**(-(regexp-5)/4)
            #reg=2*10**(-4)
            
                np.random.seed(11+13*i)
                initially_filled_traps=np.random.permutation(range(num_traps))[range(num_atoms)]
                particle_matrix=np.zeros([dim_out,dim_out])
                particle_matrix[grid[initially_filled_traps][:,0],grid[initially_filled_traps][:,1]]=1
                for k in range(num_atoms):
                    Atoms[k]=Atom_In_Trap(Strontium88(), Traps[initially_filled_traps[k]], "ground_state", spin_basis,spin_basis_labels,motional_basis,motional_basis_labels,Temp,interacting_states)
                    if (not(k==0)) and sim_traps==True:
                        Atoms[k].update_polarizability(Atoms[0].polarizability)
                        Atoms[k].update_trap_frequencies(Atoms[0].trap_frequencies)
                        Atoms[k].update_pos([Atoms[k].trap_site.x,Atoms[k].trap_site.y])
                    else:
                        Atoms[k].calc_polarizability()
                        Atoms[k].calc_trap_frequencies()
                        Atoms[k].update_pos([Atoms[k].trap_site.x,Atoms[k].trap_site.y])
                    
                #print('ROB Intialising')
                ROB=RouteOptimizerBrowaeys(Atoms, template, grid, particle_matrix,move_pickup_array[move_pickup][0],move_pickup_array[move_pickup][1])
                ROB.fill_from_center(1)
                ROB.template=ROB.template[ROB.order_of_execution]
                ROB.plot_situation_begin()
        
                #Uncomment to run Hungarian
                ROB.calc_costs(2,"lp_adjusted", lambda_adjusted, dim_in, np.mean(template,0)) #lp factor=2, lambda_adjusted, dim_in, np.mean(template,0)
                st = time.time()
                for l in range(it_create_assignment):
                    q=ROB.create_assignment("Hungarian")
                et = time.time()
                elapsed_h = (et-st)*1000
                #print('Execution time create_assignment Hungarian:', elapsed_h/it_create_assignment, 'miliseconds')
                
                #ch =  ROB.cost_of_assignment()
                #print('Cost of assignment Hungarian: ', ch)
                ROB.plot_situation_begin()
                #ROB.execute_reordering(order="fill_from_center",move_coll="no_coll",draw=1)
                #testls[i,lambda_adjusted10] = ROB.total_time()[0]+ROB.total_time()[1]
                #count+=(ROB.total_time()[0]+ROB.total_time()[1])/10
                #print("lambda: ", lambda_adjusted, "    it seed: ", i, 11+13*i, "    total time pickup: ", ROB.total_time()[0], "    total time move: ", ROB.total_time()[1])
                #print("Hungarian", "    lambda_adjusted: ", lambda_adjusted, "    it_seed: ", i, 11+13*i, "    dim_in: ", dim_in, "    total_time_AOM: ", ROB.total_time()[0]+ROB.total_time()[1])
                #print("lambda_adjusted: ", lambda_adjusted, "    it_seed: ", i, "move_time:", move_pickup_array[move_pickup][0], "pickup_time:", move_pickup_array[move_pickup][1])
        #print(count)
# =============================================================================
#                 plot_data[i,lambda_adjusted10-10,0]=lambda_adjusted
#                 plot_data[i,lambda_adjusted10-10,1]=ROB.total_time()[0]+ROB.total_time()[1]
#             plt.figure()
#             plt.plot(plot_data[i,:,0],plot_data[i,:,1])
#             plt.title("AOM time over lambda")
#             plt.xlabel('Lambda adjusted')
#             plt.ylabel('AOM total time in μs')
#             plt.show()
#         file_name = "plot_data_" + str(move_pickup_array[move_pickup][0]) + "m-" + str(move_pickup_array[move_pickup][1]) + "p_lambda=1-0.1-10.txt"
#         print("data saved under name:   ", file_name)
#         ROB.save_data(plot_data,file_name)
# =============================================================================
        #^^^^UNCOMMENT WHEN RUNNING SIMULATIONS
        
# =============================================================================
#             #Uncomment to run sinkhorn, should approximate Hungarian
#             ROB.calc_costs(2,"lp_adjusted", lambda_adjusted, dim_in, np.mean(template,0))
#             sinkhorntype = "BvN_max_alpha"      #BvN_max_alpha or max
#             #reg = 0.00001
#             st = time.time()
#             for l in range(it_create_assignment):
#                 q=ROB.create_assignment("Sinkhorn", reg, sinkhorntype)
#             et = time.time()
#             elapsed_s = (et-st)*1000
#             #print('Execution time create_assignment Sinkhorn:', elapsed_s/it_create_assignment, 'miliseconds')
#                     
#             #cs = ROB.cost_of_assignment()
#             ROB.plot_situation()
#             ROB.execute_reordering(order="fill_from_center",move_coll="no_coll",draw=1)
#             print("Sinkhorn, type: ", sinkhorntype, "    lambda_adjusted: ", lambda_adjusted, "    reg: ", reg, "    it seed: ", i, 11+13*i, "    dim in: ", dim_in, "    total time pickup: ", ROB.total_time()[0], "    total time move: ", ROB.total_time()[1], "    execution time ms:", elapsed_s/it_create_assignment)
#             #print('Total time of assignment Sinkhorn: ', ROB.total_time()[0]+ROB.total_time()[1])
#             #ROB.plot_situation()
# =============================================================================
            
        
                #Uncomment to run Browaeys and get exectution time + plot
# =============================================================================
#                 ROB.calc_costs(2,"lp")
#                 st = time.time()
#                 for l in range(it_create_assignment):
#                     q=ROB.create_assignment("Browaeys_heuristic")
#                 et = time.time()
#                 elapsed_b = (et-st)*1000
#                 #print('Execution time create_assignment Browaeys_heuristic:', elapsed_b/it_create_assignment, 'miliseconds')
#                 
#                 cb = ROB.cost_of_assignment()
#                 #print('Cost of assignment Browaeys_heuristic: ', cb)
#                 #ROB.plot_situation()
#                 ROB.execute_reordering(order="shortest_first",move_coll="no_coll",draw=1)
#                 #print("Browaeys total time: ", ROB.total_time())
#                 #print("    it seed: ", i, 11+13*i,"Browaeys total time: ", ROB.total_time()[0]+ROB.total_time()[1])
#                 count+=(ROB.total_time()[0]+ROB.total_time()[1])/10
#         print(count)
# =============================================================================
        
        
        #print('Assignment created')
        
        
        #print(q)
        #print(ROB.cost_of_assignment())
        #ROB.plot_situation()
        #time.sleep(0)
        
        #ROB.execute_reordering(order="shortest_first",move_coll="no_coll",draw=1)       
        
        # ROB=RouteOptimizerBrowaeys(particles, template,grid)
        # print(ROB.calc_costs(2,"lp"))
        
        # ROB.create_assignment("Sinkhorn")
        # print(ROB.cost_of_assignment())



