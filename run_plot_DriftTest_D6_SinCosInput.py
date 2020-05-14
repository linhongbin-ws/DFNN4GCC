from regularizeTool import EarlyStopping
from trainTool import train
from loadDataTool import load_train_N_validate_data
from os.path import join
from evaluateTool import *
import scipy.io as sio
from os import mkdir
from loadModel import get_model, load_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from AnalyticalModel import *
import numpy as np
import os

Ref_Index = 36

fk_model = FK_MODEL()
################################################################################################################
save_pdf_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual",'result')
# define train and test path
drift_test_result_path_lst = [join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual","archive","result1"),
                              join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual","archive","result2")]

D = 6
mean_arr_lst_lst = []
std_arr_lst_lst = []
max_arr_lst_lst = []
min_arr_lst_lst = []
file_name_lst = ["analytical_model", "ReLU_Dual_UDirection_BP", "ReLU_Dual_UDirection_PKD"]
for drift_test_result_path in drift_test_result_path_lst:
    cartes_dirft_mat_lst = []
    mean_arr_lst = []
    std_arr_lst = []
    max_arr_lst = []
    min_arr_lst = []
    for file_name in file_name_lst:
        drift_pos_tensor = sio.loadmat(join(drift_test_result_path,file_name))['drift_pos_tensor']
        drift_pos_cnt_arr = sio.loadmat(join(drift_test_result_path,file_name))['drift_pos_cnt_arr']
        sample_num = drift_pos_tensor.shape[2]
        drift_mat = np.zeros((sample_num, D+2 ))
        for i in range(sample_num):
            start_jnt_arr = drift_pos_tensor[-1,:,i]
            k = int(drift_pos_tensor.shape[0] - drift_pos_cnt_arr[0,i])
            end_jnt_arr = drift_pos_tensor[k,:,i]
            drift_mat[i,:D] = np.degrees(end_jnt_arr - start_jnt_arr)

            T_start = fk_model.forward(np.append(start_jnt_arr, 0))
            T_end = fk_model.forward(np.append(end_jnt_arr, 0))
            d = fk_model.transDiff(T_start, T_end)
            theta = fk_model.rotDiff(T_start, T_end)
            drift_mat[i,D] = d*1000
            drift_mat[i, D+1] = np.degrees(theta)

        # print(drift_mat)
        abs_drift_mat = np.abs(drift_mat)
        mean =np.mean(abs_drift_mat, axis=0)
        std = np.std(abs_drift_mat, axis=0)
        mean_arr_lst.append(np.mean(abs_drift_mat, axis=0))
        std_arr_lst.append(np.std(abs_drift_mat, axis=0))
        max_arr_lst.append(np.amax(abs_drift_mat, axis=0))
        min_arr_lst.append(np.amin(abs_drift_mat, axis=0))


    # print(mean_arr_lst[-1])
    mean_arr_lst_lst.append(mean_arr_lst)
    std_arr_lst_lst.append(std_arr_lst)
    max_arr_lst_lst.append(max_arr_lst)
    min_arr_lst_lst.append(max_arr_lst)



mean_arr_result_lst = []
std_arr_result_lst = []
max_arr_result_lst = []
min_arr_result_lst = []


for i in range(3):
    mean_result = np.zeros(mean_arr_lst_lst[0][0].shape)
    std_result = np.zeros(std_arr_lst_lst[0][0].shape)
    max_result = np.zeros(max_arr_lst_lst[0][0].shape)
    min_result = np.zeros(min_arr_lst_lst[0][0].shape)

    for j in range(len(drift_test_result_path_lst)):
        mean_result += mean_arr_lst_lst[j][i]
        std_result += std_arr_lst_lst[j][i]
        max_result = np.maximum(max_result, max_arr_lst_lst[j][i])
        min_result = np.minimum(min_result, min_arr_lst_lst[j][i])
    mean_result = mean_result / len(drift_test_result_path_lst)
    std_result = std_result / len(drift_test_result_path_lst)
    print(file_name_lst[i]+' mean of drift:',mean_result)
    print(file_name_lst[i] + ' std of drift:', std_result)
    print(file_name_lst[i]+' max of drift:',max_result)
    print(file_name_lst[i] + ' min of drift:', min_result)
    mean_arr_result_lst.append(mean_result)
    std_arr_result_lst.append(std_result)
    max_arr_result_lst.append(max_result)
    min_arr_result_lst.append(min_result)

# for i in range(len(mean_arr_lst)):
#     mean_arr_lst[i] = np.append(mean_arr_lst[i], np.mean(mean_arr_lst[i]))
#
#print(mean_arr_lst)




paperFontSize = 14

legend_list = ['PTM in ['+str(Ref_Index)+']', 'FDNNs with LfS', 'FDNNs with PKD']

#import matplotlib as mpl
#mpl.style.use('seaborn')
jnt_index = np.arange(1, 9)
fig, ax = plt.subplots(figsize=(7.5, 4))
w = 0.2
space = 0.2
capsize = 2
fontsize = 30
fill_color_list = ['tab:orange', 'tab:green','tab:blue']

for i in range(len(mean_arr_result_lst)):
    ax.bar(jnt_index+space*(i-1), (mean_arr_result_lst[i]).tolist(), yerr=[(mean_arr_result_lst[i]-min_arr_result_lst[i]).tolist(),
                                                                         (-mean_arr_result_lst[i]+max_arr_result_lst[i]).tolist()],
           width=w,align='center', color=fill_color_list[i], alpha=0.8, ecolor='black', capsize=capsize, label=legend_list[i])

ax.set_xticks(jnt_index)
labels = ['Joint '+str(i+1) for i in range(6)]
labels.extend([r'Tip${}^T$',r'Tip${}^{R}$'])
# labels.extend(['Tip^R',r'Tip${}_{R}$'])
# ax.set_title('Absolute RMSE for Trajectory Test')
ax.yaxis.grid(True)

# maxValue = max([max(list) for list in abs_rms_list])

maxValue = max([max(max_arr_result.tolist()) for max_arr_result in max_arr_result_lst])
csfont = {'fontname':'Times New Roman', 'fontsize':paperFontSize}

# Save the figure and show
ax.set_xticklabels(labels,**csfont)
ax.set_ylabel(r'Drift (Deg/mm)',**csfont, )

a = plt.gca()
a.set_yticklabels(a.get_yticks(), **csfont)

plt.yscale('log',basey=10)

ax.margins(y=.1, x=.03)

plt.rcParams["font.family"] = "Times New Roman"

font = matplotlib.font_manager.FontProperties(family='Times New Roman',size=paperFontSize)
ax.legend(loc='upper center', prop=font, bbox_to_anchor=(0.5, 1.18),
          fancybox=True, shadow=True, ncol=3)

plt.xticks(fontsize=paperFontSize)
plt.yticks(fontsize=paperFontSize)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)

plt.tight_layout()
plt.show()

if not os.path.exists(save_pdf_path):
    os.makedirs(save_pdf_path)


fig.savefig(join(save_pdf_path,'DriftTest_all.pdf'),bbox_inches='tight')

#
#
# jnt_index = np.arange(1,8)
# fig, ax = plt.subplots()
# w = 0.2
# space = 0.2
# capsize = 2
# fontsize = 30
#
# for i in range(len(rel_rms_list)):
#     ax.bar(jnt_index+space*(i-1), rel_rms_list[i],  width=w,align='center', color=fill_color_list[i], alpha=0.6, ecolor='black', capsize=capsize, label=legend_list[i])
#
# ax.set_xticks(jnt_index)
# labels = ['Joint '+str(i+1) for i in range(6)]
# labels.append('Avg')
# # ax.set_title('Absolute RMSE for Trajectory Test')
# ax.yaxis.grid(True)
# ax.autoscale(tight=True)
# maxValue = max([max(list) for list in rel_rms_list])
# plt.ylim(0, maxValue*1.2)
#
# # Save the figure and show
# ax.set_xticklabels(labels, fontsize=paperFontSize)
# ax.set_ylabel(r'$\epsilon_{rms}\%$', fontsize=paperFontSize)
# ax.legend(fontsize=paperFontSize)
# plt.xticks(fontsize=paperFontSize)
# plt.yticks(fontsize=paperFontSize)
# plt.tight_layout()
# plt.show()
# fig.savefig(join(train_data_path, "result",'TrajTest_RelRMS.pdf'),bbox_inches='tight')
#
#
# print('Avg Absolute RMSE: ',[lst[-1] for lst in abs_rms_list])
# print('Avg Relative RMSE: ',[lst[-1] for lst in rel_rms_list])
