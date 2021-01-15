


from model_psa_o2 import *

numcores = multiprocessing.cpu_count()
numcores


numcores = multiprocessing.cpu_count() - 1
separations = [3 for i in range(500)] + [3.5 for i in range(500)] + [4 for i in range(500)] + [5 for i in range(500)] +  [7 for i in range(500)] + [10 for i in range(500)] +  [12 for i in range(500)] +  [14 for i in range(500)] +  [18 for i in range(500)]



time_simulation=3000
presentation_period = 250
angle_separation=30

target_onset = 700
target_offset = target_onset + presentation_period
distractor_onset = 300
distractor_offset = distractor_onset + presentation_period
order2 = distractor_onset<target_onset ##boolean


r = model1(totalTime=time_simulation,  presentation_period=presentation_period,  angle_separation=angle_separation, 
           targ_onset=target_onset, dist_onset=distractor_onset,  order_2=order2,
           tauE=60, tauI=10, tauf=7000, taud=80, 
           I0I=0.4, U=0.4,
           GEE=0.022, GEI=0.019, GIE=0.01 , GII=0.1,
           sigE=0.2, sigI=0.04,
           kappa_E=100, kappa_I=1.5, k_inhib=0.07, kappa_stim=50,
           N=512, plot_connectivity=False, plot_dyniamic=True, plot_heatmap=False, plot_fit=False, save_RE=True)

print('Interference= ' +str(r[0]))





print('1_02')
outputs_1_02 = Parallel(n_jobs = numcores)(delayed(model)(totalTime=6200, 
                                                          targ_onset = 200,
                                                          dist_onset=550, 
                                                          presentation_period=250,
                                                          separation=sep, 
                                                          inhib_curr=False, 
                                                          time_ex_input=0, 
                                                          sigE=1.2,
                                                          GEE=0.016,
                                                          plot_connectivity=False, 
                                                          plot_dyniamic=False, 
                                                          plot_heatmap=False, 
                                                          plot_fit=False ) for sep in separations) 
