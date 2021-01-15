


from model_psa_o2 import *

numcores = multiprocessing.cpu_count()-3
numcores


numcores = multiprocessing.cpu_count() - 1
separations = [3 for i in range(500)] + [3.5 for i in range(500)] + [4 for i in range(500)] + [5 for i in range(500)] +  [7 for i in range(500)] + [10 for i in range(500)] +  [12 for i in range(500)] +  [14 for i in range(500)] +  [18 for i in range(500)]


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
