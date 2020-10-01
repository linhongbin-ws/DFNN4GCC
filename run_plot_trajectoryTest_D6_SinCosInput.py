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
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', type=str, required=True, help="MTML or MTMR")
    parser.add_argument('--sn', type=str, required=True, help="Serial Number: e.g. 28002")
    parser.add_argument('--sample_num', type=int, required=True, help="sampling number of testing data points")
    args = parser.parse_args()
    ARM_NAME = args.arm
    SN = args.sn
    sample_num = args.sample_num


    Ref_Index = 36
    ################################################################################################################


    # define train and test path
    train_data_path = join(".","data", ARM_NAME + '_' + SN, "real", "uniform", "N4", 'D6_SinCosInput', "dual")
    test_data_path = join(".","data", ARM_NAME + '_' + SN, "real", "random", 'N'+str(sample_num),'D6_SinCosInput')

    # load Trajectory Test experiment data
    test_dataset = load_data_dir(join(test_data_path, "data"), device='cpu',input_scaler=None, output_scaler=None,
                                 is_inputScale = False, is_outputScale = False)

    test_input_mat = test_dataset.x_data.numpy()
    test_ouput_mat = test_dataset.y_data.numpy()


    test_output_hat_mat_List = []
    legend_list = []
    #
    # # get predict CAD Model output
    # MTM_CAD_model = MTM_CAD()
    # test_output_hat_mat_List.append(MTM_CAD_model.predict(test_input_mat))
    # legend_list.append('CAD')

    # get predict MLSE4POL Model output
    MTM_MLSE4POL_Model = MTM_MLSE4POL()
    test_output_hat_mat_List.append(MTM_MLSE4POL_Model.predict(test_input_mat))



    device = 'cpu'
    D = 6
    # get predict DNN with Knowledge Distillation output
    use_net = 'ReLU_Dual_UDirection'
    train_type = 'BP'
    load_model_path = join(train_data_path, "result", "model")
    model = get_model('MTM', use_net, D, device=device)
    model, _, _ = load_model(load_model_path, use_net+'_'+train_type, model)
    test_output_hat_mat_List.append(model.predict_NP(test_input_mat))



    # get predict DNN with Knowledge Distillation output
    use_net = 'ReLU_Dual_UDirection'
    train_type = 'PKD'
    load_model_path = join(train_data_path, "result", "model")
    model = get_model('MTM', use_net, D, device=device)
    model, _, _ = load_model(load_model_path, use_net+'_'+train_type, model)
    test_output_hat_mat_List.append(model.predict_NP(test_input_mat))



    legend_list = ['PTM in ['+str(Ref_Index)+']', 'FDNNs with LfS', 'FDNNs with PKD']


    # plot predict error bar figures
    abs_rms_list = []
    rel_rms_list = []
    mean_rel_rms_list = []
    for i in range(len(test_output_hat_mat_List)):
        err_output_mat = np.abs(test_output_hat_mat_List[i] - test_ouput_mat)
        abs_rms_list.append(np.sqrt(np.mean(np.square(err_output_mat), axis=0)).tolist())
        rel_rms_list.append(np.sqrt(np.divide(np.mean(np.square(err_output_mat), axis=0),
                                        np.mean(np.square(test_ouput_mat), axis=0))).tolist())

    for i in range(len(rel_rms_list)):
        abs_rms_list[i].append(np.mean(abs_rms_list[i], axis=0))
        rel_rms_list[i].append(np.mean(rel_rms_list[i],axis=0))

    for i in range(len(rel_rms_list)):
        rel_rms_list[i] =[k*100 for k in rel_rms_list[i]]

    #print(err_output_mat)



    paperFontSize = 16
    jnt_index = np.arange(1,8)
    # fig, ax = plt.subplots()
    # # matplotlib.rc('text', usetex=True)
    # # matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    #
    # plt.rcParams["font.family"] = "Times New Roman"
    # w = 0.2
    # space = 0.2
    # capsize = 2
    # fontsize = 30
    # fill_color_list = ['tab:green', 'tab:orange', 'tab:blue']
    #
    # for i in range(len(abs_rms_list)):
    #     ax.bar(jnt_index+space*(i-1), abs_rms_list[i],  width=w,align='center', color=fill_color_list[i], alpha=0.8, ecolor='black', capsize=capsize, label=legend_list[i])
    #
    # ax.set_xticks(jnt_index)
    # labels = ['Joint '+str(i+1) for i in range(6)]
    # labels.append('Avg')
    # # ax.set_title('Absolute RMSE for Trajectory Test')
    # ax.yaxis.grid(True)
    # ax.autoscale(tight=True)
    # # maxValue = max([max(list) for list in abs_rms_list])
    # # plt.ylim(0, maxValue*1.2)
    # ax.margins(y=.1, x=.03)
    #
    # # Save the figure and show
    # csfont = {'fontname':'Times New Roman', 'fontsize':paperFontSize}
    # ax.set_xticklabels(labels, **csfont)
    # ax.set_ylabel(r'$\epsilon_{rms}$ (N.m)', **csfont)
    # a = plt.gca()
    # a.set_yticklabels(a.get_yticks(), **csfont)
    # ax.legend(fontsize=paperFontSize)
    # plt.xticks(fontsize=paperFontSize)
    # plt.yticks(fontsize=paperFontSize)
    #
    #
    # plt.tight_layout()
    # plt.show()
    # fig.savefig(join(train_data_path, "result",'TrajTest_AbsRMS.pdf'),bbox_inches='tight')



    jnt_index = np.arange(1,8)
    fig, ax = plt.subplots(figsize=(8, 4))
    fill_color_list = ['tab:green', 'tab:orange', 'tab:blue']

    plt.rcParams["font.family"] = "Times New Roman"
    w = 0.2
    space = 0.2
    capsize = 2
    fontsize = 30

    for i in range(len(rel_rms_list)):
        ax.bar(jnt_index+space*(i-1), rel_rms_list[i],  width=w,align='center', color=fill_color_list[i], alpha=0.8, ecolor='black', capsize=capsize, label=legend_list[i])

    ax.set_xticks(jnt_index)
    labels = ['Joint '+str(i+1) for i in range(6)]
    labels.append('Avg')
    # ax.set_title('Absolute RMSE for Trajectory Test')
    ax.yaxis.grid(True)
    ax.autoscale(tight=True)
    # maxValue = max([max(list) for list in rel_rms_list])
    # plt.ylim(0, maxValue*1.2)
    ax.margins(y=.1, x=.03)

    # Save the figure and show
    csfont = {'fontname':'Times New Roman', 'fontsize':paperFontSize}
    ax.set_xticklabels(labels, **csfont)
    ax.set_ylabel(r'$\epsilon_{rms}$% (N.m)',  **csfont)
    a = plt.gca()
    a.set_yticklabels(a.get_yticks(), **csfont)

    font = matplotlib.font_manager.FontProperties(family='Times New Roman',size=paperFontSize)
    ax.legend(loc='upper center', prop=font, bbox_to_anchor=(0.5, 1.2),
              fancybox=True, shadow=True, ncol=3)
    plt.xticks(fontsize=paperFontSize)
    plt.yticks(fontsize=paperFontSize)
    plt.tight_layout()
    plt.show()
    fig.savefig(join(train_data_path, "result",'TrajTest_RelRMS.pdf'),bbox_inches='tight')


    print('Avg Absolute RMSE: ',[lst[-1] for lst in abs_rms_list])
    print('Avg Relative RMSE: ',[lst[-1] for lst in rel_rms_list])

    output_list = [lst[-1] for lst in rel_rms_list]
    print("latex:")
    print("{:.1f} &".format(output_list[0]), end=' ')
    print("{:.1f} &".format(output_list[1]), end=' ')
    print("{:.1f} &".format(output_list[2]), end=' ')