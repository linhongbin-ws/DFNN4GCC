from regularizeTool import EarlyStopping
from trainTool import train
from loadDataTool import load_train_N_validate_data
from os.path import join
from evaluateTool import *
import scipy.io as sio
from os import mkdir
from loadModel import get_model, load_model
import matplotlib
# matplotlib.use('MacOSX')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from AnalyticalModel import *
import numpy as np
from pathlib import Path



################################################################################################################

# define train and test path
# train_data_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual")
# test_data_path = join("data", "MTMR_28002", "real", "random", 'N319','D6_SinCosInput')

# load Trajectory Test experiment data


def cal_baselines_rms(train_data_path, test_data_path):
    test_dataset = load_data_dir(join(test_data_path, "data"), device='cpu',input_scaler=None, output_scaler=None,
                                 is_inputScale = False, is_outputScale = False)

    test_input_mat = test_dataset.x_data.numpy()
    test_ouput_mat = test_dataset.y_data.numpy()


    test_output_hat_mat_List = []
    legend_list = []


    # get predict MLSE4POL Model output
    analytical_model = MTM_CAD()
    test_output_hat_mat_List.append(analytical_model.predict(test_input_mat))
    legend_list.append('CAD')


    device = 'cpu'
    D = 6
    # get predict DNN with Knowledge Distillation output
    use_net = 'ReLU_Dual_UDirection'
    train_type = 'BP'
    load_model_path = join(train_data_path, "result", "model")
    model = get_model('MTM', use_net, D, device=device)
    model, _, _ = load_model(load_model_path, use_net+'_'+train_type, model)
    test_output_hat_mat_List.append(model.predict_NP(test_input_mat))
    legend_list.append('DFNN with LfS')


    # get predict DNN with Knowledge Distillation output
    use_net = 'ReLU_Dual_UDirection'
    train_type = 'PKD'
    load_model_path = join(train_data_path, "result", "model")
    model = get_model('MTM', use_net, D, device=device)
    model, _, _ = load_model(load_model_path, use_net+'_'+train_type, model)
    test_output_hat_mat_List.append(model.predict_NP(test_input_mat))
    legend_list.append('DFNN with PKD')



    # plot predict error bar figures
    abs_rms_list = []
    rel_rms_list = []
    mean_rel_rms_list = []
    for i in range(len(test_output_hat_mat_List)):
        err_output_mat = np.abs(test_output_hat_mat_List[i] - test_ouput_mat)
        abs_rms_list.append(np.sqrt(np.mean(np.square(err_output_mat), axis=0)).tolist())
        rel_rms_list.append(np.sqrt(np.divide(np.mean(np.square(err_output_mat), axis=0),
                                        np.mean(np.square(test_ouput_mat), axis=0))).tolist())

    abs_rms_mean_list = []
    abs_rms_std_list = []
    rel_rms_mean_list = []
    rel_rms_std_list = []


    for i in range(len(rel_rms_list)):
        abs_rms_mean_list.append(np.mean(abs_rms_list[i], axis=0))
        abs_rms_std_list.append(np.std(abs_rms_list[i], axis=0))
        rel_rms_mean_list.append(np.mean(rel_rms_list[i], axis=0)*100)
        rel_rms_std_list.append(np.std(rel_rms_list[i], axis=0)*100)

    # print(abs_rms_mean_list)
    # print(rel_rms_mean_list)

    return abs_rms_mean_list, rel_rms_mean_list




# train_simulate_num_list = [10, 50, 100,500,1000, 5000]

train_simulate_num_list = [10, 50, 100,500,1000, 5000]
repetitive_num = 4
# DistScale = 0.02
DistScale = 1
baseline_num = 3
font_size = 20
legend_size = 17

abs_rms_mean_arr_list = []
rel_rms_mean_arr_list = []
abs_rms_std_arr_list = []
rel_rms_std_arr_list = []

for i in range(len(train_simulate_num_list)):
    abs_rms_mean_mat = np.zeros((repetitive_num, baseline_num))
    rel_rms_mean_mat = np.zeros((repetitive_num, baseline_num))

    for j in range(repetitive_num):
        train_data_path = join("data", "MTMR_28002", "sim", 'random', 'Dist_'+str(DistScale), 'train', "N"+str(train_simulate_num_list[i]), 'D6_SinCosInput',
                               str(j+1))
        test_data_path = join("data", "MTMR_28002", "sim", 'random', 'Dist_'+str(DistScale), 'test', "N20000", 'D6_SinCosInput')
        abs_rms_mean_list, rel_rms_mean_list = cal_baselines_rms(train_data_path, test_data_path)
        abs_rms_mean_mat[j,:] = np.asarray(abs_rms_mean_list)
        rel_rms_mean_mat[j, :] = np.asarray(rel_rms_mean_list)

    abs_rms_mean_arr = np.mean(abs_rms_mean_mat, axis=0)
    rel_rms_mean_arr = np.mean(rel_rms_mean_mat, axis=0)
    abs_rms_std_arr = np.std(abs_rms_mean_mat, axis=0)
    rel_rms_std_arr = np.std(rel_rms_mean_mat, axis=0)

    abs_rms_mean_arr_list.append(abs_rms_mean_arr)
    rel_rms_mean_arr_list.append(rel_rms_mean_arr)
    abs_rms_std_arr_list.append(abs_rms_std_arr)
    rel_rms_std_arr_list.append(rel_rms_std_arr)


fig,ax = plt.subplots()

legend_list = ['Physical Teacher Model', 'DFNN with LfS', 'DFNN with PKD']
fill_color_list = ['tab:blue','tab:orange', 'tab:green']
for i in range(baseline_num):
    x = train_simulate_num_list
    y = [abs_rms_mean_arr[i] for abs_rms_mean_arr in abs_rms_mean_arr_list]
    y_err = [abs_rms_std_arr[i] for abs_rms_std_arr in abs_rms_std_arr_list]

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    y_err_arr = np.asarray(y_err)
    plt.plot(x_arr, y_arr, '-', color= fill_color_list[i])
    plt.fill_between(x_arr, y_arr-y_err_arr, y_arr+y_err_arr, alpha=0.5, facecolor=fill_color_list[i], label=legend_list[i])

plt.legend(loc='upper right',fontsize=legend_size)
plt.xlabel(r'$T^{s}$', fontsize=font_size)
plt.ylabel(r'$\epsilon_{rms}$', fontsize=font_size)
plt.xscale('log')
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.tick_params(axis='both', which='minor', labelsize=font_size)

plt.yticks(fontsize=font_size)
plt.tight_layout()
ax.yaxis.grid(True)
# ax.autoscale(tight=True)

plt.show()
save_dir = join("data", "MTMR_28002", "sim", 'random', 'Dist_'+str(DistScale), 'train', "result")
Path(save_dir).mkdir(parents=True, exist_ok=True)
fig.savefig(join(save_dir,'Dist_'+str(DistScale)+'_OfflineTest_AbsRMS.pdf'),bbox_inches='tight')




fig,ax = plt.subplots()

for i in range(baseline_num):
    x = train_simulate_num_list
    y = [rel_rms_mean_arr[i] for rel_rms_mean_arr in rel_rms_mean_arr_list]
    y_err = [rel_rms_std_arr[i] for rel_rms_std_arr in rel_rms_std_arr_list]

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    y_err_arr = np.asarray(y_err)
    plt.plot(x_arr, y_arr, '-', color= fill_color_list[i])
    plt.fill_between(x_arr, y_arr-y_err_arr, y_arr+y_err_arr, alpha=0.5, facecolor=fill_color_list[i], label=legend_list[i])

plt.legend(loc='upper right',fontsize=legend_size)

plt.xlabel(r'$T^{s}$', fontsize=font_size)
plt.ylabel(r'$\epsilon_{rms}\%$', fontsize=font_size)
plt.xscale('log')
ax.tick_params(axis='both', which='major', labelsize=font_size)
ax.tick_params(axis='both', which='minor', labelsize=font_size)

plt.yticks(fontsize=font_size)
plt.tight_layout()


plt.show()
save_dir = join("data", "MTMR_28002", "sim", 'random', 'Dist_'+str(DistScale), 'train', "result")
Path(save_dir).mkdir(parents=True, exist_ok=True)
fig.savefig(join(save_dir,'Dist_'+str(DistScale)+'_OfflineTest_RelRMS.pdf'),bbox_inches='tight')


#print(err_output_mat)
#
#
#
# jnt_index = np.arange(1,8)
# fig, ax = plt.subplots()
# w = 0.2
# space = 0.2
# capsize = 2
# fontsize = 30
#
# for i in range(len(abs_rms_list)):
#     ax.bar(jnt_index+space*(i-1), abs_rms_list[i],  width=w,align='center', alpha=0.5, ecolor='black', capsize=capsize, label=legend_list[i])
#
# ax.set_xticks(jnt_index)
# labels = ['Joint '+str(i+1) for i in range(6)]
# labels.append('Avg')
# # ax.set_title('Absolute RMSE for Trajectory Test')
# ax.yaxis.grid(True)
# ax.autoscale(tight=True)
# maxValue = max([max(list) for list in abs_rms_list])
# plt.ylim(0, maxValue*1.2)
#
# # Save the figure and show
# ax.set_xticklabels(labels, fontsize=font_size)
# ax.set_ylabel(r'$\epsilon_{rms}$', fontsize=font_size)
# ax.legend(fontsize=font_size)
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
# plt.tight_layout()
# plt.show()
# fig.savefig(join(train_data_path, "result",'TrajTest_AbsRMS.pdf'),bbox_inches='tight')
#
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
#     ax.bar(jnt_index+space*(i-1), rel_rms_list[i],  width=w,align='center', alpha=0.5, ecolor='black', capsize=capsize, label=legend_list[i])
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
# ax.set_xticklabels(labels, fontsize=font_size)
# ax.set_ylabel(r'$\epsilon_{rms}\%$', fontsize=font_size)
# ax.legend(fontsize=font_size)
# plt.xticks(fontsize=font_size)
# plt.yticks(fontsize=font_size)
# plt.tight_layout()
# plt.show()
# fig.savefig(join(train_data_path, "result",'TrajTest_RelRMS.pdf'),bbox_inches='tight')
#
#
# print('Avg Absolute RMSE: ',[lst[-1] for lst in abs_rms_list])
# print('Avg Relative RMSE: ',[lst[-1] for lst in rel_rms_list])