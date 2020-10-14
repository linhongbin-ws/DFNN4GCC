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
import scipy.io as sio
from pathlib import Path
import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42


################################################################################################################

# define train and test path
# train_data_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual")
# test_data_path = join("data", "MTMR_28002", "real", "random", 'N319','D6_SinCosInput')

# load Trajectory Test experiment data

Ref_Index = 30

def cal_baselines_rms(train_data_path, test_data_path, TM_param_vec, BP_train_data_path):
    test_dataset = load_data_dir(join(test_data_path, "data"), device='cpu',input_scaler=None, output_scaler=None,
                                 is_inputScale = False, is_outputScale = False)

    test_input_mat = test_dataset.x_data.numpy()
    test_ouput_mat = test_dataset.y_data.numpy()


    test_output_hat_mat_List = []
    legend_list = []

    teacherModel = MTM_MLSE4POL()
    teacherModel.param_vec = TM_param_vec


    test_output_hat_mat_List.append(teacherModel.predict(test_input_mat))
    legend_list.append('CAD')


    device = 'cpu'
    D = 6
    # get predict DNN with Knowledge Distillation output
    use_net = 'ReLU_Dual_UDirection'
    train_type = 'BP'
    load_model_path = join(BP_train_data_path, "result", "model")
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



train_simulate_num_list = [10, 50, 100,500,1000, 5000]
repetitive_num = 2
simulate_type = 'MLSE4POL'
param_noise_scale_lst = [1e-3, 4e-3]

baseline_num = 3
font_size = 20
legend_size = 17


for k in range(len(param_noise_scale_lst)):
    abs_rms_mean_arr_list = []
    rel_rms_mean_arr_list = []
    abs_rms_std_arr_list = []
    rel_rms_std_arr_list = []
    root_path = join("data", "MTMR_28002", "sim", 'random', simulate_type, "bias_"+str(param_noise_scale_lst[k]))
    BP_root_path = join("data", "MTMR_28002", "sim", 'random', simulate_type, "bias_"+str(param_noise_scale_lst[1]))
    load_dict = sio.loadmat(join(root_path, 'simulation_param.mat'))
    TM_param_vec = load_dict['TM_param_vec']
    for i in range(len(train_simulate_num_list)):
        abs_rms_mean_mat = np.zeros((repetitive_num, baseline_num))
        rel_rms_mean_mat = np.zeros((repetitive_num, baseline_num))

        for j in range(repetitive_num):
            train_data_path = join(root_path, 'train', "N"+str(train_simulate_num_list[i]), 'D6_SinCosInput',str(j+1))
            BP_train_data_path = join(BP_root_path, 'train', "N"+str(train_simulate_num_list[i]), 'D6_SinCosInput',str(j+1))
            test_data_path = join(root_path, 'test', "N20000", 'D6_SinCosInput')
            abs_rms_mean_list, rel_rms_mean_list = cal_baselines_rms(train_data_path, test_data_path, TM_param_vec, BP_train_data_path)
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

#
# fig,ax = plt.subplots()

    # fig,ax = plt.subplots()
    #
    # legend_list = ['Physical Teacher Model', 'DFNN with LfS', 'DFNN with PKD']
    # fill_color_list = ['tab:blue','tab:orange', 'tab:green']
    # for i in range(baseline_num):
    #     x = train_simulate_num_list
    #     y = [abs_rms_mean_arr[i] for abs_rms_mean_arr in abs_rms_mean_arr_list]
    #     y_err = [abs_rms_std_arr[i] for abs_rms_std_arr in abs_rms_std_arr_list]
    #
    #     x_arr = np.asarray(x)
    #     y_arr = np.asarray(y)
    #     y_err_arr = np.asarray(y_err)
    #     plt.plot(x_arr, y_arr, '-', color= fill_color_list[i])
    #     plt.fill_between(x_arr, y_arr-y_err_arr, y_arr+y_err_arr, alpha=0.5, facecolor=fill_color_list[i], label=legend_list[i])
    #
    # plt.legend(loc='upper right',fontsize=legend_size)
    # plt.xlabel(r'$T^{s}$', fontsize=font_size)
    # plt.ylabel(r'$\epsilon_{rms}$', fontsize=font_size)
    # plt.xscale('log')
    # ax.tick_params(axis='both', which='major', labelsize=font_size)
    # ax.tick_params(axis='both', which='minor', labelsize=font_size)
    #
    # plt.yticks(fontsize=font_size)
    # plt.tight_layout()
    # ax.yaxis.grid(True)
    # # ax.autoscale(tight=True)
    #
    # plt.show()
    # save_dir = join("data", "MTMR_28002", "sim", 'random', 'Dist_'+str(DistScale), 'train', "result")
    # Path(save_dir).mkdir(parents=True, exist_ok=True)
    # fig.savefig(join(save_dir,'Dist_'+str(DistScale)+'_OfflineTest_AbsRMS.pdf'),bbox_inches='tight')
    #
    if k == 0:
        legend_list = ['Low-bias PTM in ['+str(Ref_Index)+']', 'FDNNs with LfS', 'FDNNs with PKD']
    else:
        legend_list = ['High-bias PTM in ['+str(Ref_Index)+']', 'FDNNs with LfS', 'FDNNs with PKD']

    fill_color_list = ['tab:green', 'tab:orange', 'tab:blue']
    paperFontSize = 20

    fig,ax = plt.subplots(figsize=(8.4, 4.8))

    plt.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams.update({'font.size': paperFontSize})
    for i in range(baseline_num):
        x = train_simulate_num_list
        y = [rel_rms_mean_arr[i] for rel_rms_mean_arr in rel_rms_mean_arr_list]
        y_err = [rel_rms_std_arr[i] for rel_rms_std_arr in rel_rms_std_arr_list]

        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        y_err_arr = np.asarray(y_err)
        plt.plot(x_arr, y_arr, '-', color= fill_color_list[i])
        plt.fill_between(x_arr, y_arr-y_err_arr, y_arr+y_err_arr, alpha=0.5, facecolor=fill_color_list[i], label=legend_list[i])

    font = matplotlib.font_manager.FontProperties(family='Times New Roman', size=paperFontSize)
    # ax.legend(loc='upper center', prop=font, bbox_to_anchor=(0.5, 1),
    #           fancybox=True, shadow=True, ncol=2)
    plt.legend(bbox_to_anchor=(0., 1.06, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.,fancybox=True, shadow=True)


    # Save the figure and show
    csfont = {'fontname': 'Times New Roman', 'fontsize': paperFontSize}
    # plt.xlabel(r'$T^{s}$', fontsize=font_size)
    # plt.ylabel(r'$\epsilon_{rms}\%$', fontsize=font_size)
    ax.set_xlabel(r'$T^{s}$', **csfont)
    ax.set_ylabel(r'$\epsilon_{rms}\%$', **csfont)

    a = plt.gca()
    a.set_xticklabels(a.get_xticks(), **csfont)
    a.set_yticklabels(a.get_yticks(), **csfont)

    plt.xscale('log')
    # ax.tick_params(axis='both', which='major', labelsize=font_size)
    # ax.tick_params(axis='both', which='minor', labelsize=font_size)
    ax.margins(y=.1, x=.03)


    plt.tight_layout()


    plt.show()
    save_dir = join(root_path, 'train', "result")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams['pdf.fonttype'] = 42
    fig.savefig(join('.','Dist_'+simulate_type+'_'+str(param_noise_scale_lst[k])+'_OfflineTest_RelRMS.pdf'),bbox_inches='tight')




