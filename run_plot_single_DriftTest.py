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
import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['pdf.fonttype'] = 42
fk_model = FK_MODEL()
################################################################################################################
save_pdf_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual",'result')
# define train and test path
drift_test_result_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual","archive","result1")
D = 6
sample_idx = 20
file_name_lst = ["analytical_model", "ReLU_Dual_UDirection_BP", "ReLU_Dual_UDirection_PKD"]

Ref_Index = 30

d_arr_lst = []
theta_arr_lst = []
drift_time_lst = []
for file_name in file_name_lst:
    drift_pos_tensor = sio.loadmat(join(drift_test_result_path,file_name))['drift_pos_tensor']
    drift_pos_cnt_arr = sio.loadmat(join(drift_test_result_path,file_name))['drift_pos_cnt_arr']
    drift_time = sio.loadmat(join(drift_test_result_path, file_name))['drift_time']
    sample_num = int(drift_pos_cnt_arr[0][sample_idx])
    drift_mat = drift_pos_tensor[drift_pos_tensor.shape[0]-sample_num:,:,sample_idx]
    drift_mat = np.flip(drift_mat, axis=0)
    d_arr = np.zeros(sample_num)
    theta_arr = np.zeros(sample_num)

    T_start = fk_model.forward(np.append(drift_mat[0,:], 0))
    for j in range(1,sample_num):
        T_end = fk_model.forward(np.append(drift_mat[j, :], 0))
        d = fk_model.transDiff(T_start, T_end)
        theta = fk_model.rotDiff(T_start, T_end)
        d_arr[j] = d*1000
        theta_arr[j] = np.degrees(theta)


    d_arr_lst.append(d_arr)
    theta_arr_lst.append(theta_arr)
    drift_time_lst.append(drift_time[0][sample_idx])






paperFontSize = 14
paperLineWidth =2.5

legend_list = ['PTM in ['+str(Ref_Index)+']', 'FDNNs with LfS', 'FDNNs with PKD']


fig, ax = plt.subplots(2, figsize=(6.7, 4))
fill_color_list = ['tab:green','tab:orange','tab:blue']

for i in range(3):
    x = np.linspace(0,drift_time_lst[i], d_arr_lst[i].shape[0])
    ax[0].plot(x.tolist(), d_arr_lst[i].tolist(),color=fill_color_list[i], alpha=0.8, label=legend_list[i],linewidth=paperLineWidth)
    ax[1].plot(x.tolist(), theta_arr_lst[i].tolist(), color=fill_color_list[i], alpha=0.8, label=legend_list[i],linewidth=paperLineWidth)

ax[0].yaxis.grid(True)
ax[1].yaxis.grid(True)

# maxValue = max([max(list) for list in abs_rms_list])


csfont = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'fontsize'   : paperFontSize}
# Save the figure and show
ax[1].set_xlabel('time (s)',**csfont)
ax[0].set_ylabel(r'Drift (mm)',**csfont)
ax[1].set_ylabel(r'Drift (Deg)',**csfont)
plt.axes(ax[0])
a = plt.gca()
a.set_xticklabels(a.get_xticks(), **csfont)
a.set_yticklabels(a.get_yticks(), **csfont)
plt.axes(ax[1])
a = plt.gca()
a.set_xticklabels(a.get_xticks(), **csfont)
a.set_yticklabels(a.get_yticks(), **csfont)

# plt.yscale('log',basey=10)

ax[0].margins(y=.1, x=.03)
ax[1].margins(y=.1, x=.03)

font = matplotlib.font_manager.FontProperties(family='Times New Roman',size=paperFontSize)
ax[0].legend(loc='upper center', prop=font, bbox_to_anchor=(0.5, 1.42),
          fancybox=True, shadow=True, ncol=3)

ax[0].tick_params(axis='both', which='major', labelsize=paperFontSize)
ax[0].tick_params(axis='both', which='minor', labelsize=paperFontSize)
ax[1].tick_params(axis='both', which='major', labelsize=paperFontSize)
ax[1].tick_params(axis='both', which='minor', labelsize=paperFontSize)
# ax[0].set_tick_params(labelsize=paperFontSize)
# ax[1].set_tick_params(labelsize=paperFontSize)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False)
plt.axes(ax[0])
plt.tight_layout()
plt.axes(ax[1])
plt.tight_layout()
plt.show()
#
if not os.path.exists(save_pdf_path):
    os.makedirs(save_pdf_path)


#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42

fig.savefig(join('.','DriftTest_single.pdf'),bbox_inches='tight')

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
