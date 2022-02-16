# -*- coding: utf-8 -*-
"""
@author: David Bestue
"""

import math
from math import floor, exp, sqrt, pi
import cmath
import numpy
from numpy import e, cos, zeros, arange, roll, where, random, ones, mean, reshape, dot, array, flipud, pi, exp, dot, angle, degrees, shape, linspace
import matplotlib.pyplot as plt
from itertools import chain
import scipy
from scipy import special
import numpy as np 
import seaborn as sns
import time
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scipy.signal
from scipy.optimize import curve_fit 
from scipy import stats





###############
############### Accesory functions
###############


def decode_rE(rE, a_ini=0, a_fin=360, N=512):
    #Population vector for a given rE
    # return ( angle in radians, absolut angle in radians, abs angle in degrees )
    N=len(rE)
    rE_f = np.flipud(rE)
    Angles = np.linspace(a_ini, a_fin, N) 
    angles=np.radians(Angles)
    rE_f = np.reshape(rE_f, (1,N))
    R = numpy.sum(np.dot(rE_f,exp(1j*angles)))/numpy.sum(rE_f) ## finding it with imagianry numbers
    angle_decoded = np.degrees(np.angle(R))
    if angle_decoded<0:
        angle_decoded = 360+angle_decoded
    
    return angle_decoded



def model_I0E_constant(value, N=512):
    y=[value for x in range(N)]
    return np.reshape(np.array(y), (N,1))



###############
############### MODEL STPA function
###############


def model_STPA(totalTime, targ_onset, presentation_period, angle_pos=180,    
    tauE=60, tauI=10, tauf=7000, taud=80, I0I=0.4, U=0.4,
    GEE=0.016, GEI=0.015, GIE=0.012 , GII=0.007, sigE=0.06, sigI=0.04,
    kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=20,
    N=512, plot_connectivity=True, plot_dyniamic=True, save_RE=True):
    ##
    #Temporal and spatial settings
    st_sim =time.time()
    dt=2;
    nsteps=int(floor(totalTime/dt));
    origin = np.radians(angle_pos)
    targ_offset = targ_onset + presentation_period;
    targon = floor(targ_onset/dt);
    targoff = floor(targ_offset/dt);
    ######
    ###### Connectivitiess
    v_E=np.zeros((N));
    v_I=np.zeros((N));
    WE=np.zeros((N,N));
    WI=np.zeros((N,N));
    theta = [float(range(0,N)[i])/N*2*pi for i in range(0,N)];
    for i in range(0, N):
        v_E_new=[e**(kappa_E*cos(theta[f]))/(2*pi*scipy.special.i0(kappa_E)) for f in range(0, len(theta))]    
        v_I_new=[e**(kappa_I*cos(theta[f]))/(2*pi*scipy.special.i0(kappa_I)) + k_inhib for f in range(0, len(theta))] 
        #    
        vE_NEW=np.roll(v_E_new,i)
        vI_NEW=np.roll(v_I_new,i) 
        # 
        WE[:,i]=vE_NEW
        WI[:,i]=vI_NEW
        #   
    ###### Stimuli
    target=np.zeros((N))
    for i in range(0, N):
        target[i]=e**(kappa_stim*cos(theta[i] + origin ))  / (2*pi*scipy.special.i0(kappa_stim)) #   
    #
    noise_stim =  0 #np.random.normal(0, 0.01, N)
    target = target+ noise_stim
    target=reshape(target, (N,1)) 
    #
    ###### Empty arrays of the variables 
    mf=1
    rE=np.zeros((N,1));
    rI=np.zeros((N,1)); 
    u = np.ones((N,1))*U
    x = np.ones((N,1))
    RE=np.zeros((N,nsteps));
    RI=np.zeros((N,nsteps));
    p_u=np.ones((N,nsteps));
    p_x=np.ones((N,nsteps));
    #
    ## Different quadrant_selectivity options gaussian
    I0E_open =  0.7 
    I0E_close= 0.1 
    quadrant_selectivity_close = model_I0E_constant(I0E_close)
    quadrant_selectivity_open =  model_I0E_constant(I0E_open)
    ## state before the simulation strats (depending on the cue)
    quadrant_selectivity = quadrant_selectivity_open
    ###
    ###
    ### currents duing the simulation 
    f = lambda x : x*x*(x>0)*(x<1) + reshape(array([cmath.sqrt(4*x[i]-3) for i in range(0, len(x))]).real, (N,1)) * (x>=1)
    for i in range(0, nsteps):
        noiseE = sigE*random.randn(N,1);
        noiseI = sigI*random.randn(N,1);
        #differential equations for current
        IE= GEE*dot(WE, (rE/mf*u*x)) - GIE*dot(WI,rI/mf) + quadrant_selectivity;
        II= GEI*dot(WE,rE/mf) +  (I0I-GII*mean(rI/mf))*ones((N,1));
        #
        ## presentation stims 
        if i>targon and i<targoff:
            IE=IE+target;
            II=II+target;
        #
        #####################################################
        #####################################################
        #rates 
        rE = rE*mf + (f(IE)*mf - rE*mf + noiseE)*dt/tauE;
        rI = rI*mf + (f(II)*mf - rI*mf + noiseI)*dt/tauI;
        ### formulas for synaptic plasticity: paper mongillo 2008
        u = u + ((U - u) / tauf + U*(1-u)*(rE/mf)/1000)*dt;
        x = x + ((1 - x)/taud - u*x*(rE/mf)/1000)*dt;
        rEr=np.reshape(rE, N)*10
        rIr=np.reshape(rI, N)
        ur=np.reshape(u, N)
        xr=np.reshape(x, N)
        #append
        RE[:,i] = rEr;
        RI[:,i] = rIr;
        p_u[:,i] = ur;
        p_x[:,i] = xr;   

        rE=rE/mf 
        rI=rI/mf 
    #
    #### Decode position
    final_position_bump = decode_rE(rE)
    #
    #### Plots 
    # Plot of the connectivity profile
    if plot_connectivity ==True:
        fig=plt.figure()
        fig.tight_layout()
        fig.set_size_inches(13, 4)
        fig.add_subplot(121)
        plt.plot(WE[250, :], color='darkorange', label='E')
        plt.plot(WI[250, :], color='lightblue', label = 'I')
        plt.ylim(0,6)
        plt.gca().spines['right'].set_visible(False) #no right axis
        plt.gca().spines['top'].set_visible(False) #no  top axis
        plt.gca().get_xaxis().tick_bottom()
        plt.gca().get_yaxis().tick_left()
        plt.title('Connectivity WE & WI')
        #
        fig.add_subplot(122)
        plt.plot(WE[250, :] - WI[250, :] , 'darkred', label='E-I')
        plt.gca().spines['right'].set_visible(False) #no right axis
        plt.gca().spines['top'].set_visible(False) #no  top axis
        plt.gca().get_xaxis().tick_bottom()
        plt.gca().get_yaxis().tick_left()
        plt.title('Effective connectivity')
        plt.show(block=False)
    #
    # Plot dynamics
    p_targ = int((N * np.degrees(origin))/360)
    if plot_dyniamic==True:        
        fig = plt.figure()
        fig.tight_layout()
        fig.set_size_inches(13, 4)
        fig.add_subplot(121)
        p_targ = int((N * np.degrees(origin))/360)
        plt.title('Synaptic dynamics for target')
        plt.plot(np.flipud(p_u)[p_targ, :], color='darkblue', label='prob. release')
        plt.plot(np.flipud(p_x)[p_targ, :], color='darkred', label='pool vesicles')
        plt.xlabel('time (ms)')
        plt.xticks(np.linspace(0, int(totalTime/dt), 7), np.array(np.linspace(0, int(totalTime/dt), 7)*2, dtype=int))        
        plt.gca().spines['right'].set_visible(False)  # aesthetics                                                                              # remove right spines
        plt.gca().spines['top'].set_visible(False)                                                                                  # remove top spines
        plt.gca().get_xaxis().tick_bottom()                                                                                         
        plt.gca().get_yaxis().tick_left()
        plt.gca().tick_params(direction='in') #direction
        plt.legend(frameon=False)
        #
        fig.add_subplot(122)
        plt.title('Rate dynamics')
        plt.plot(np.flipud(RE)[p_targ, :], color='darkgreen', label='target')
        plt.xlabel('time (ms)')
        plt.xticks(np.linspace(0, int(totalTime/dt), 7), np.array(np.linspace(0, int(totalTime/dt), 7)*2, dtype=int)) 
        plt.ylabel('rate (Hz)')
        plt.gca().spines['right'].set_visible(False)  # aesthetics                                                                              # remove right spines
        plt.gca().spines['top'].set_visible(False)                                                                                  # remove top spines
        plt.gca().get_xaxis().tick_bottom()                                                                                         
        plt.gca().get_yaxis().tick_left()
        plt.gca().tick_params(direction='in') #direction
        plt.legend(frameon=False)
        plt.show(block=False)
    #
    #   
    ### Output
    ###return bias_target, bias_dist, number_of_bumps, angle_separation, RE #rE[p_targ][0], I0E
    if save_RE==True:
        return final_position_bump, RE
    else:
        return final_position_bump
    



###############
############### Plot heatmap function
###############


def heatmap_model_STPA(RE, time_simulation, angle_pos, target_onset, pres_period, save_name=False):
    #pal_cyan = sns.color_palette("viridis")
    dims=np.shape(RE)
    dimN = dims[0]
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(RE, cmap="cividis", vmin=0, vmax=45,  cbar=True, 
                cbar_kws={"shrink": .82, 'ticks' : [0, 15, 30, 45], 'label': 'rate (Hz)'})
    ax.figure.axes[-1].yaxis.label.set_size(20)
    ax.figure.axes[-1].tick_params(labelsize=20)
    ax.axis('tight')
    plt.gca().set_ylabel('')
    plt.gca().set_xlabel('')
    plt.gca().set_title('')
    p_stim = angle_pos * (dims[0]/360)
    #
    stimon = target_onset/2
    stimoff = (target_onset + pres_period) / 2
    #
    plt.gca().set_xticks([])
    plt.gca().set_xticklabels([])
    #
    plt.gca().set_yticks([0, int(dimN/4), int(dimN/2),  int(3*dimN/4), int(dimN) ])
    plt.gca().set_yticklabels(['360','','180', '', '0'], fontsize=20)
    #
    plt.gca().set_xlabel('', fontsize=20);
    plt.gca().set_ylabel('neuron preferred ($^\circ$)', fontsize=20);
    plt.gca().set_ylim(dimN+60, -45)
    ###
    ##line stims 
    c1='darkorange' 
    c2='k'
    s1on=stimon
    s1off=stimoff
    ##
    plt.plot([0, s1on], [-15, -15], linestyle='-', color='k', linewidth=2)
    plt.plot([s1on, s1on], [-15, -40], linestyle='-', color=c1, linewidth=2)
    plt.plot([s1on, s1off], [-40, -40], linestyle='-', color=c1, linewidth=2)
    plt.plot([s1off, s1off], [-15, -40], linestyle='-', color=c1, linewidth=2)
    plt.plot([s1off, dims[1]], [-15, -15], linestyle='-', color='k', linewidth=2)
    #
    #time
    x1sec = 1000 * dims[1] / time_simulation
    plt.plot([dims[1]-x1sec, dims[1]], [dimN+30, dimN+30], 'k-', linewidth=2)
    plt.text(dims[1]-300, 600, '1s', fontsize=20);
    if save_name!=False:
        plt.savefig(save_name + '.png', transparent=True) ##to save it transparent
    plt.show(block=False)



#####
##### Codes to run simulations
#####

time_simulation=3000 #ms
Angle_pres = 270 #degrees
presentation_period = 250 #ms
target_onset = 300 #ms
target_offset = target_onset + presentation_period


results= model_STPA(totalTime=time_simulation,  presentation_period=presentation_period,   
           targ_onset=target_onset, angle_pos=Angle_pres,
           tauE=60, tauI=10, tauf=7000, taud=80, 
           I0I=0.4, U=0.4,
           GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1,
           sigE=2.8, sigI=2.2,
           kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=50,
           N=512, plot_connectivity=True, plot_dyniamic=True)



print('Decoded position: ' + str(np.round(results[0], 3)))


heatmap_model_STPA(RE=results[1], time_simulation=time_simulation, 
                   angle_pos=Angle_pres, target_onset=target_onset, 
                   pres_period=presentation_period, save_name=False)