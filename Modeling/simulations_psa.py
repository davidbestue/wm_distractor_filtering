

#### Call the functions and run the simulations

from model_psa import *

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


print('1_7')
outputs_1_7 = Parallel(n_jobs = numcores)(delayed(model)(totalTime=6200, 
                                                          targ_onset = 200,
                                                          dist_onset=3900, 
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




print('2_02')
outputs_2_02 = Parallel(n_jobs = numcores)(delayed(model)(totalTime=6200,
                                                          targ_onset = 550,
                                                          dist_onset=200, 
                                                          presentation_period=250,
                                                          separation=sep, 
                                                          inhib_curr=True, 
                                                          time_ex_input=0, 
                                                          sigE=1.2, 
                                                          plot_connectivity=False, 
                                                          plot_dyniamic=False, 
                                                          plot_heatmap=False, 
                                                          plot_fit=False ) for sep in separations) 




print('2_7')
outputs_2_7 = Parallel(n_jobs = numcores)(delayed(model)(totalTime=9600,
                                                          targ_onset = 3950,
                                                          dist_onset=200, 
                                                          presentation_period=250,
                                                          separation=sep, 
                                                          inhib_curr=True, 
                                                          time_ex_input=0, 
                                                          sigE=1.2, 
                                                          plot_connectivity=False, 
                                                          plot_dyniamic=False, 
                                                          plot_heatmap=False, 
                                                          plot_fit=False ) for sep in separations) 



df_1_02 = pd.DataFrame(outputs_1_02)
df_1_02.columns= ['bias_target', 'bias_dist', 'number_of_bumps', 'angle_separation']
df_1_02['delay']=0.2
df_1_02['order'] =1

df_1_7 = pd.DataFrame(outputs_1_7)
df_1_7.columns= ['bias_target', 'bias_dist', 'number_of_bumps', 'angle_separation']
df_1_7['delay']=7
df_1_7['order'] =1

df_2_02 = pd.DataFrame(outputs_2_02)
df_2_02.columns= ['bias_target', 'bias_dist', 'number_of_bumps', 'angle_separation']
df_2_02['delay']=0.2
df_2_02['order'] =2

df_2_7 = pd.DataFrame(outputs_2_7)
df_2_7.columns= ['bias_target', 'bias_dist', 'number_of_bumps', 'angle_separation']
df_2_7['delay']=7
df_2_7['order'] =2

res_simulations = pd.concat([df_1_02, df_1_7, df_2_02, df_2_7 ], ignore_index=True)


#path_save = '/home/david/Desktop/res_sim.xlsx'
#res_simulations.to_excel(path_save)
### after that ypiu scp to your local and proceed with the analysis (and plot the results)
