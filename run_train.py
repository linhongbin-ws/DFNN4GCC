from __future__ import print_function
from regularizeTool import EarlyStopping
from trainTool import train,KDtrain
from loadDataTool import load_preProcessData
from os.path import join
from evaluateTool import *
import scipy.io as sio
from os import makedirs
from loadModel import get_model, save_model
from HyperParam import get_hyper_param
from AnalyticalModel import *
import scipy
import sys
import argparse


def loop_func(train_data_path, test_data_path, use_net, robot, train_type='BP', valid_data_path=None, is_sim = False, is_inputNormalized=True, is_outputNormalized=True,
              sim_distScale=None, simulation_param_path=None, load_PTM_param_file_str=None):
    param_dict = get_hyper_param(robot, train_type=train_type, is_sim=is_sim, sim_distScale = sim_distScale)

    max_training_epoch = param_dict['max_training_epoch'] # stop train when reach maximum training epoch
    goal_loss = param_dict['goal_loss'] # stop train when reach goal loss
    batch_size = param_dict['batch_size'] # batch size for mini-batch gradient descent
    weight_decay = param_dict['weight_decay']
    device = param_dict['device']
    earlyStop_patience = param_dict['earlyStop_patience']
    learning_rate = param_dict['learning_rate']
    D = param_dict['D']

    device = torch.device(device)
    model = get_model('MTM', use_net, D, device=device)
    if train_type == 'BP':
        train_loader, valid_loader, _, input_mean, input_std, output_mean, output_std =load_preProcessData(join(train_data_path, "data"),
                                                                                                           batch_size,
                                                                                                           device,
                                                                                                           valid_ratio=param_dict['valid_ratio'],
                                                                                                           valid_data_path=join(valid_data_path, "data") if valid_data_path is not None else None)
    elif train_type == 'PKD':
        if not is_sim:
            teacherModel = MTM_MLSE4POL()
        else:
            teacherModel = MTM_MLSE4POL()
            load_dict = sio.loadmat(join(simulation_param_path, 'simulation_param.mat'))
            teacherModel.param_vec = load_dict['TM_param_vec']

        if load_PTM_param_file_str is not None:
            teacherModel.decode_json_file(load_PTM_param_file_str)

        train_loader, valid_loader, teacher_loader, input_mean, input_std, output_mean, output_std = load_preProcessData(join(train_data_path, "data"),
                                                                                                                        batch_size,
                                                                                                                        device,
                                                                                                                        valid_ratio=param_dict['valid_ratio'],
                                                                                                                        valid_data_path=join(valid_data_path, "data") if valid_data_path is not None else None,
                                                                                                                        teacherModel=teacherModel,
                                                                                                                        teacher_sample_num=param_dict['teacher_sample_num'],
                                                                                                                        is_inputNormalized=is_inputNormalized,
                                                                                                                        is_outputNormalized=is_outputNormalized)


    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)

    ### Train model
    model.set_normalized_param(input_mean, input_std, output_mean, output_std)
    if train_type=='BP':
        model, train_losses, valid_losses = train(model, train_loader, valid_loader, optimizer, loss_fn, early_stopping, max_training_epoch, goal_loss, is_plot=False)
    elif train_type == 'PKD':
        model, train_losses, valid_losses = KDtrain(model, train_loader, valid_loader, teacher_loader, optimizer, loss_fn, early_stopping,
                        max_training_epoch, goal_loss, param_dict['initLamda'], param_dict['endLamda'], param_dict['decayStepsLamda'], is_plot=False)
    else:
        raise Exception("cannot recoginze the train type")

    # save model to "result/model" folder
    test_dataset = load_data_dir(join(test_data_path, "data"), device='cpu', input_scaler=None, output_scaler=None, is_inputScale = False, is_outputScale = False)
    feature_mat = test_dataset.x_data.numpy()
    target_mat = test_dataset.y_data.numpy()
    model = model.to('cpu')
    target_hat_mat = model.predict_NP(feature_mat)

    rel_rms_vec = np.sqrt(np.divide(np.mean(np.square(target_hat_mat - target_mat), axis=0),
                                    np.mean(np.square(target_mat), axis=0)))

    abs_rms_vec = np.sqrt(np.mean(np.square(target_hat_mat - target_mat), axis=0))

    print('Absolute RMS for each joint are:', abs_rms_vec)
    print('Relative RMS for each joint are:', rel_rms_vec)


    model_save_path = join(train_data_path,"result","model")
    try:
        makedirs(model_save_path)
    except:
        print('Make directory: ', model_save_path + " already exist")

    if is_inputNormalized and is_outputNormalized:
        save_file_name = use_net + '_' + train_type
    elif is_inputNormalized and not is_outputNormalized:
        save_file_name = use_net + '_' + train_type + '_noOutNorm'
    elif not is_inputNormalized and is_outputNormalized:
        save_file_name = use_net + '_' + train_type + '_noInNorm'
    else:
        save_file_name = use_net + '_' + train_type + '_noInOutNorm'

    save_model(model_save_path, save_file_name, model)

    learning_curve_path = join(train_data_path,"result")
    save_file_name = use_net + '_' + train_type +'_learnCurve.mat'
    scipy.io.savemat(join(learning_curve_path, save_file_name), {'train_losses': train_losses,
                                                                 'valid_losses': valid_losses})

    return abs_rms_vec, rel_rms_vec

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--arm', type=str, required=True, help="MTML or MTMR")
    parser.add_argument('--sn', type=str, required=True, help="Serial Number: e.g. 28002")
    args = parser.parse_args()

    ARM_NAME = args.arm
    SN = args.sn

    if not (ARM_NAME == 'MTML' or ARM_NAME == 'MTMR'):
        print('The first argument(arm)  should be either MTML or MTMR')
        sys.exit()

    if not (len(SN)==5):
        print('The second argument(sn) should be a string with five number, e.g.: 28008')
        sys.exit()

    # #############################################################
    #
    load_PTM_param_file_str = join("data",  ARM_NAME+'_'+SN, "real", "gc-"+ARM_NAME+"-"+SN +".json")
    train_data_path = join(".", "data", ARM_NAME+'_'+SN, "real", "uniform", "N4", 'D6_SinCosInput', "dual")
    valid_data_path = join(".", "data", ARM_NAME+'_'+SN, "real", "random",  "N160", 'D6_SinCosInput')
    test_data_path = join(".", "data", ARM_NAME+'_'+SN, "real", "random", "N40",'D6_SinCosInput')


    ###  train models
    abs_rms_vec_dict = dict()
    rel_rms_vec_dict = dict()
    abs_rms_vec_dict['LFS'], rel_rms_vec_dict['LFS'] = loop_func(train_data_path, test_data_path, 'ReLU_Dual_UDirection','MTM', train_type='BP', valid_data_path=valid_data_path)
    abs_rms_vec_dict['PKD'], rel_rms_vec_dict['PKD'] = loop_func(train_data_path, test_data_path, 'ReLU_Dual_UDirection','MTM', train_type='PKD', valid_data_path= valid_data_path, load_PTM_param_file_str = load_PTM_param_file_str)
    print("")
    print("")
    print("")
    print("")
    print("")

    print('===========train result==============')
    print('ARMSE(Absolute Root Mean Square Error) and RMSE(Relative Root Mean Square Error) for Joint 1 to Joint 6 for {}_{}'.format(ARM_NAME, SN))
    keys = abs_rms_vec_dict.keys()
    for key in keys:
        print('--------------------------------')
        print('{} :'.format(key))
        for i in range(len(abs_rms_vec_dict[key])):
            print("Joint {}: ARMSE({:.4f}), RRMSE({:.2f} %)".format(i+1, abs_rms_vec_dict[key][i], rel_rms_vec_dict[key][i]*100))

        # print('Average for 6 Joints: ARMSE({:.4f}), RRMSE({:.2f} %)'.format(i+1, abs_rms_vec_dict[key].mean(), rel_rms_vec_dict[key].mean()*100))
