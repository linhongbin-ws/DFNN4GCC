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
import time
import datetime
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
    #
    # ### Get the predict output from test data and save to Matlab file
    # train_dataset = load_data_dir(join(train_data_path,'data'), device='cpu', input_scaler=None, output_scaler=None, is_inputScale = False, is_outputScale = False)
    # train_input_mat = train_dataset.x_data
    # train_output_mat = train_dataset.y_data
    # model = model.to('cpu')
    # test_dataset = load_data_dir(join(test_data_path,"data"), device='cpu', input_scaler=None, output_scaler=None, is_inputScale = False, is_outputScale = False)
    # test_input_mat = test_dataset.x_data
    # test_output_mat = predict(model, test_input_mat, input_scaler, output_scaler)
    # try:
    #     mkdir(join(train_data_path,"result"))
    # except:
    #     print('Make directory: ', join(train_data_path,"result") + " already exist")
    #
    # # save data as .mat file
    # save_result_path = join(train_data_path, "result", use_net+'.mat')
    # print('Save result: ', save_result_path)
    # sio.savemat(save_result_path, {'test_input_mat': test_input_mat.numpy(),
    #                               'test_output_mat': test_output_mat.numpy(),
    #                               'train_input_mat': train_input_mat.numpy(),
    #                               'train_output_mat': train_output_mat.numpy()})
    #
    # test_loss, abs_rms_vec, rel_rms_vec = evaluate_rms(model, loss_fn, test_data_path, input_scaler, output_scaler, device, verbose=True)

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



################################################################################################################
# uncomment to use

#################################################################################
# train real MTM

### setting
ARM_NAME = "MTML"
SN = "41878"
########

load_PTM_param_file_str = join("data",  ARM_NAME+'_'+SN, "real", "gc-"+ARM_NAME+"-"+SN +".json")
train_data_path = join(".", "data", ARM_NAME+'_'+SN, "real", "uniform", "N4", 'D6_SinCosInput', "dual")
valid_data_path = join(".", "data", ARM_NAME+'_'+SN, "real", "random",  "N160", 'D6_SinCosInput')
test_data_path = join(".", "data", ARM_NAME+'_'+SN, "real", "random", "N40",'D6_SinCosInput')




###  train models
loop_func(train_data_path, test_data_path, 'ReLU_Dual_UDirection','MTM', train_type='BP', valid_data_path=valid_data_path)
loop_func(train_data_path, test_data_path, 'ReLU_Dual_UDirection','MTM', train_type='PKD', valid_data_path= valid_data_path, load_PTM_param_file_str = load_PTM_param_file_str)
# loop_func(train_data_path, test_data_path, 'ReLU_Dual_UDirection','MTMR28002', train_type='PKD', valid_data_path= valid_data_path, is_inputNormalized=False, is_outputNormalized=True)
# loop_func(train_data_path, test_data_path, 'ReLU_Dual_UDirection','MTMR28002', train_type='PKD', valid_data_path= valid_data_path, is_inputNormalized=True, is_outputNormalized=False)
# loop_func(train_data_path, test_data_path, 'ReLU_Dual_UDirection','MTMR28002', train_type='PKD', valid_data_path= valid_data_path, is_inputNormalized=False, is_outputNormalized=False)


#
# ##################################################################################
# # train sim MTM
# sum_start_time = time.clock()
#
# train_simulate_num_list = [10, 50, 100,500,1000, 5000]
# validate_simulate_num = 20000
# test_simulate_num = 20000
#
# repetitive_num = 10
# param_noise_scale_lst = [1e-3, 4e-3]
# for k in range(len(param_noise_scale_lst)):
#
#     save_dir = join("data", "MTMR_28002", "sim", "random", 'MLSE4POL', "bias_"+str(param_noise_scale_lst[k]))
#
#     test_data_path = join(save_dir, 'validate', 'N'+str(validate_simulate_num), 'D6_SinCosInput')
#     valid_data_path = join(save_dir, 'test', 'N'+str(validate_simulate_num), 'D6_SinCosInput')
#     for j in range(len(train_simulate_num_list)):
#         for i in range(repetitive_num):
#             loop_time = time.clock()
#             print("train_simulate_num ", train_simulate_num_list[j], " repetitive no: ", i)
#             train_data_path = join(save_dir, "train", 'N'+str(train_simulate_num_list[j]), 'D6_SinCosInput', str(i+1))
#             print("train BP")
#             loop_func(train_data_path, test_data_path, 'ReLU_Dual_UDirection', 'MTMR28002', train_type='BP',is_sim=True,valid_data_path=valid_data_path,
#                       sim_distScale = param_noise_scale_lst[k])
#
#             # print the time info
#             loop_time = time.clock() - loop_time
#             sum_time = time.clock() - sum_start_time
#             total_num = len(train_simulate_num_list)*repetitive_num * len(param_noise_scale_lst)
#             finish_num = i + j*repetitive_num + k*repetitive_num*len(train_simulate_num_list) + 0.5
#             total_time = sum_time *  total_num / finish_num
#             print("")
#             print("*******************************")
#             print("finish (" + str(finish_num) + "/" + str(total_num) + ")",
#                   "duration of one loop: " + str(datetime.timedelta(seconds=loop_time)),
#                   "  time:" + str(datetime.timedelta(seconds=sum_time)) + " / " + str(
#                       datetime.timedelta(seconds=total_time)))
#             print("*******************************")
#             print("")
#
#             print("train PKD")
#             loop_func(train_data_path, test_data_path, 'ReLU_Dual_UDirection', 'MTMR28002', train_type='PKD',
#                       is_sim=True, valid_data_path=valid_data_path, sim_distScale = param_noise_scale_lst[k], simulation_param_path=save_dir)
#
#             # print the time info
#             loop_time = time.clock() - loop_time
#             sum_time = time.clock() - sum_start_time
#             total_num = len(train_simulate_num_list)*repetitive_num * len(param_noise_scale_lst)
#             finish_num = i + j * repetitive_num + k * repetitive_num * len(train_simulate_num_list) + 1
#             total_time = sum_time * total_num / finish_num
#             print("")
#             print("*******************************")
#             print("finish (" + str(finish_num) + "/" + str(total_num) + ")",
#                   "duration of one loop: " + str(datetime.timedelta(seconds=loop_time)),
#                   "  time:" + str(datetime.timedelta(seconds=sum_time)) + " / " + str(
#                       datetime.timedelta(seconds=total_time)))
#             print("*******************************")
#             print("")
#
#
#
#
#
#
#
#
#
