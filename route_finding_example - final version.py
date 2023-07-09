#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:02:01 2022

@author: robert
Edits done by Ben Esseling for the atom sorting BEP Q3-Q4 2022-2023
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

dim_in = 5
dim_out = int(np.ceil(2**(1/2)*dim_in))
num_atoms = np.int64(dim_out**2/2)

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



move_pickup  = [50,600]
lambda_adjusted = 2                 #lambda parameter in the lambda algorithm
        
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
    
# Initialize ROB
ROB=RouteOptimizerBrowaeys(Atoms, template, grid, particle_matrix,move_pickup[0],move_pickup[1])
ROB.fill_from_center(1)
ROB.template=ROB.template[ROB.order_of_execution]
ROB.plot_situation_begin()

#Calculate cost matrix, find assignments, and execute reordering
ROB.calc_costs(2,"lp_adjusted", lambda_adjusted, dim_in, np.mean(template,0)) 
ROB.create_assignment("Hungarian")
ROB.plot_situation_begin()

ROB.execute_reordering(order="fill_from_center",move_coll="no_coll",draw=1)


