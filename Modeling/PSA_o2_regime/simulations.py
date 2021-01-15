
from model_psa_o2 import *

numcores = multiprocessing.cpu_count() - 3



##################################################################### 1_0.2_close #########################################################################

n_simuls=10

time_simulation=3300
presentation_period=1500
angle_separation=70
target_onset = 300
target_offset = target_onset + presentation_period
distractor_onset = 300
distractor_offset = distractor_onset + presentation_period
order2 = distractor_onset<target_onset ##boolean


n_times = [time_simulation for i in range(n_simuls)] 

outputs_1_02_close = Parallel(n_jobs = numcores)(delayed(model1)(totalTime=t,  
    presentation_period=presentation_period,  angle_separation=angle_separation, 
    targ_onset=target_onset, dist_onset=distractor_onset,  order_2=order2, 
    tauE=60, tauI=10, tauf=7000, taud=80, I0I=0.4, U=0.4, 
    GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1, sigE=0.2, sigI=0.04, 
    kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=50, 
    N=512, plot_connectivity=False, plot_dyniamic=False, plot_heatmap=False, plot_fit=False, save_RE=False) for t in n_times) 

