# -*- coding: utf-8 -*-
"""
@author: David Bestue
"""

from model_psa_o2 import *

numcores = multiprocessing.cpu_count() - 4
n_simuls=1000


###########################################################################################################################################################
##################################################################### 1_0.2_close #########################################################################
###########################################################################################################################################################
print('1_0.2_close')

time_simulation=3300
presentation_period=200
angle_separation=30 #60

target_onset = 300
target_offset = target_onset + presentation_period
distractor_onset = 700
distractor_offset = distractor_onset + presentation_period
order2 = distractor_onset<target_onset ##boolean


n_times = [time_simulation for i in range(n_simuls)] 

outputs_1_02_close = Parallel(n_jobs = numcores)(delayed(model1)(totalTime=t,  
    presentation_period=presentation_period,  angle_separation=angle_separation, 
    targ_onset=target_onset, dist_onset=distractor_onset,  order_2=order2, 
    tauE=60, tauI=10, tauf=7000, taud=80, I0I=0.4, U=0.4, 
    GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, 
    sigE=2.8, sigI=2.2, 
    kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=50, 
    N=512, plot_connectivity=False, plot_dyniamic=False, plot_heatmap=False, plot_fit=False, save_RE=False) for t in n_times) 


df_1_02_close = pd.DataFrame(outputs_1_02_close)
df_1_02_close.columns=['interference']
df_1_02_close['order'] = 1
df_1_02_close['delay'] = 'short'
df_1_02_close['distance'] = 'close'


###########################################################################################################################################################
##################################################################### 1_0.2_far ###########################################################################
###########################################################################################################################################################
print('1_0.2_far')

time_simulation=3300
presentation_period=200
angle_separation=60

target_onset = 300
target_offset = target_onset + presentation_period
distractor_onset = 700
distractor_offset = distractor_onset + presentation_period
order2 = distractor_onset<target_onset ##boolean


n_times = [time_simulation for i in range(n_simuls)] 

outputs_1_02_far = Parallel(n_jobs = numcores)(delayed(model1)(totalTime=t,  
    presentation_period=presentation_period,  angle_separation=angle_separation, 
    targ_onset=target_onset, dist_onset=distractor_onset,  order_2=order2, 
    tauE=60, tauI=10, tauf=7000, taud=80, I0I=0.4, U=0.4, 
    GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, 
    sigE=2.8, sigI=2.2, 
    kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=50, 
    N=512, plot_connectivity=False, plot_dyniamic=False, plot_heatmap=False, plot_fit=False, save_RE=False) for t in n_times) 


df_1_02_far = pd.DataFrame(outputs_1_02_far)
df_1_02_far.columns=['interference']
df_1_02_far['order'] = 1
df_1_02_far['delay'] = 'short'
df_1_02_far['distance'] = 'far'



###########################################################################################################################################################
##################################################################### 1_7_close ###########################################################################
###########################################################################################################################################################
print('1_7_close')

time_simulation=3300
presentation_period=200
angle_separation=30

target_onset = 300
target_offset = target_onset + presentation_period
distractor_onset = 1500
distractor_offset = distractor_onset + presentation_period
order2 = distractor_onset<target_onset ##boolean


n_times = [time_simulation for i in range(n_simuls)] 

outputs_1_7_close = Parallel(n_jobs = numcores)(delayed(model1)(totalTime=t,  
    presentation_period=presentation_period,  angle_separation=angle_separation, 
    targ_onset=target_onset, dist_onset=distractor_onset,  order_2=order2, 
    tauE=60, tauI=10, tauf=7000, taud=80, I0I=0.4, U=0.4, 
    GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, 
    sigE=2.8, sigI=2.2, 
    kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=50, 
    N=512, plot_connectivity=False, plot_dyniamic=False, plot_heatmap=False, plot_fit=False, save_RE=False) for t in n_times) 


df_1_7_close = pd.DataFrame(outputs_1_7_close)
df_1_7_close.columns=['interference']
df_1_7_close['order'] = 1
df_1_7_close['delay'] = 'long'
df_1_7_close['distance'] = 'close'



###########################################################################################################################################################
##################################################################### 1_7_far ###########################################################################
###########################################################################################################################################################
print('1_7_far')

time_simulation=3300
presentation_period=200
angle_separation=60

target_onset = 300
target_offset = target_onset + presentation_period
distractor_onset = 1500
distractor_offset = distractor_onset + presentation_period
order2 = distractor_onset<target_onset ##boolean


n_times = [time_simulation for i in range(n_simuls)] 

outputs_1_7_far = Parallel(n_jobs = numcores)(delayed(model1)(totalTime=t,  
    presentation_period=presentation_period,  angle_separation=angle_separation, 
    targ_onset=target_onset, dist_onset=distractor_onset,  order_2=order2, 
    tauE=60, tauI=10, tauf=7000, taud=80, I0I=0.4, U=0.4, 
    GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, 
    sigE=2.8, sigI=2.2, 
    kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=50, 
    N=512, plot_connectivity=False, plot_dyniamic=False, plot_heatmap=False, plot_fit=False, save_RE=False) for t in n_times) 


df_1_7_far = pd.DataFrame(outputs_1_7_far)
df_1_7_far.columns=['interference']
df_1_7_far['order'] = 1
df_1_7_far['delay'] = 'long'
df_1_7_far['distance'] = 'far'


###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################



###########################################################################################################################################################
##################################################################### 2_0.2_close #########################################################################
###########################################################################################################################################################
print('2_0.2_close')

time_simulation=3700
presentation_period=200
angle_separation=30 #60

target_onset = 700
target_offset = target_onset + presentation_period
distractor_onset = 300
distractor_offset = distractor_onset + presentation_period
order2 = distractor_onset<target_onset ##boolean


n_times = [time_simulation for i in range(n_simuls)] 

outputs_2_02_close = Parallel(n_jobs = numcores)(delayed(model1)(totalTime=t,  
    presentation_period=presentation_period,  angle_separation=angle_separation, 
    targ_onset=target_onset, dist_onset=distractor_onset,  order_2=order2, 
    tauE=60, tauI=10, tauf=7000, taud=80, I0I=0.4, U=0.4, 
    GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, 
    sigE=2.8, sigI=2.2, 
    kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=50, 
    N=512, plot_connectivity=False, plot_dyniamic=False, plot_heatmap=False, plot_fit=False, save_RE=False) for t in n_times) 


df_2_02_close = pd.DataFrame(outputs_2_02_close)
df_2_02_close.columns=['interference']
df_2_02_close['order'] = 2
df_2_02_close['delay'] = 'short'
df_2_02_close['distance'] = 'close'


###########################################################################################################################################################
##################################################################### 2_0.2_far ###########################################################################
###########################################################################################################################################################
print('2_0.2_far')

time_simulation=3700
presentation_period=200
angle_separation=60

target_onset = 700
target_offset = target_onset + presentation_period
distractor_onset = 300
distractor_offset = distractor_onset + presentation_period
order2 = distractor_onset<target_onset ##boolean


n_times = [time_simulation for i in range(n_simuls)] 

outputs_2_02_far = Parallel(n_jobs = numcores)(delayed(model1)(totalTime=t,  
    presentation_period=presentation_period,  angle_separation=angle_separation, 
    targ_onset=target_onset, dist_onset=distractor_onset,  order_2=order2, 
    tauE=60, tauI=10, tauf=7000, taud=80, I0I=0.4, U=0.4, 
    GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, 
    sigE=2.8, sigI=2.2, 
    kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=50, 
    N=512, plot_connectivity=False, plot_dyniamic=False, plot_heatmap=False, plot_fit=False, save_RE=False) for t in n_times) 


df_2_02_far = pd.DataFrame(outputs_2_02_far)
df_2_02_far.columns=['interference']
df_2_02_far['order'] = 2
df_2_02_far['delay'] = 'short'
df_2_02_far['distance'] = 'far'


###########################################################################################################################################################
##################################################################### 2_7_close ###########################################################################
###########################################################################################################################################################
print('2_7_close')

time_simulation=5500
presentation_period=200
angle_separation=30

target_onset = 1500
target_offset = target_onset + presentation_period
distractor_onset = 300
distractor_offset = distractor_onset + presentation_period
order2 = distractor_onset<target_onset ##boolean


n_times = [time_simulation for i in range(n_simuls)] 

outputs_2_7_close = Parallel(n_jobs = numcores)(delayed(model1)(totalTime=t,  
    presentation_period=presentation_period,  angle_separation=angle_separation, 
    targ_onset=target_onset, dist_onset=distractor_onset,  order_2=order2, 
    tauE=60, tauI=10, tauf=7000, taud=80, I0I=0.4, U=0.4, 
    GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, 
    sigE=2.8, sigI=2.2, 
    kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=50, 
    N=512, plot_connectivity=False, plot_dyniamic=False, plot_heatmap=False, plot_fit=False, save_RE=False) for t in n_times) 


df_2_7_close = pd.DataFrame(outputs_2_7_close)
df_2_7_close.columns=['interference']
df_2_7_close['order'] = 2
df_2_7_close['delay'] = 'long'
df_2_7_close['distance'] = 'close'


###########################################################################################################################################################
##################################################################### 2_7_far ###########################################################################
###########################################################################################################################################################
print('2_7_far')

time_simulation=5500
presentation_period=200
angle_separation=60

target_onset = 1500
target_offset = target_onset + presentation_period
distractor_onset = 300
distractor_offset = distractor_onset + presentation_period
order2 = distractor_onset<target_onset ##boolean


n_times = [time_simulation for i in range(n_simuls)] 

outputs_2_7_far = Parallel(n_jobs = numcores)(delayed(model1)(totalTime=t,  
    presentation_period=presentation_period,  angle_separation=angle_separation, 
    targ_onset=target_onset, dist_onset=distractor_onset,  order_2=order2, 
    tauE=60, tauI=10, tauf=7000, taud=80, I0I=0.4, U=0.4, 
    GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, 
    sigE=2.8, sigI=2.2, 
    kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=50, 
    N=512, plot_connectivity=False, plot_dyniamic=False, plot_heatmap=False, plot_fit=False, save_RE=False) for t in n_times) 


df_2_7_far = pd.DataFrame(outputs_2_7_far)
df_2_7_far.columns=['interference']
df_2_7_far['order'] = 2
df_2_7_far['delay'] = 'long'
df_2_7_far['distance'] = 'far'




###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################
###########################################################################################################################################################


df=pd.concat([df_1_02_close, df_1_02_far, df_1_7_close, df_1_7_far, 
              df_2_02_close, df_2_02_far, df_2_7_close, df_2_7_far])



#
df.to_excel('results_psa_o2.xlsx')
#
