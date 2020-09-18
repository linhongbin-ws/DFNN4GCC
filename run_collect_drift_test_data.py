from __future__ import print_function
import time
import scipy.io
import datetime
from Controller import Controller
import numpy as np
from os.path import join
import scipy.io
import os
import argparse
import sys
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', type=str, required=True, help="MTML or MTMR")
    parser.add_argument('--sn', type=str, required=True, help="Serial Number: e.g. 28002")
    parser.add_argument('--sample_num', type=int, required=True, help="sampling number of testing data points")
    args = parser.parse_args()

    # Hyper params
    MTM_ARM = args.arm
    SN = args.sn
    sample_num = args.sample_num
    save_testing_point_path = join("data", MTM_ARM+'_'+SN, "real", "dirftTest", "N4", 'D6_SinCosInput', "dual")
    joint_limit_json_path =   join("data", MTM_ARM+'_'+SN, "real", 'dataCollection_config_customized.json')
    load_model_path =         join("data", MTM_ARM+'_'+SN, "real", "uniform",   "N4", 'D6_SinCosInput', "dual", "result", "model")
    save_result_path =        join("data", MTM_ARM+'_'+SN, "real", "dirftTest", "N4", 'D6_SinCosInput', "dual", "result")
    load_PTM_param_path =     join("data", MTM_ARM+'_'+SN, "real")
    D = 6
    use_net = 'ReLU_Dual_UDirection'
    verbose_silent_level = 1


    # initialize controller
    controller = Controller(MTM_ARM)
    controller.verbose_silent_level = verbose_silent_level
    if not controller.load_jointLimit_json(joint_limit_json_path):
        sys.exit()

    # generate random jnt configurations for Drift Test
    q_mat, ready_q_mat = controller.random_testing_configuration(sample_num)
    if not os.path.exists(save_testing_point_path):
        os.makedirs(save_testing_point_path)
    scipy.io.savemat(join(save_testing_point_path, 'testing_points.mat'), {'q_mat': q_mat,
                                 'ready_q_mat': ready_q_mat})

    test_controller_list = ['PTM', 'LFS', 'PKD']
    print(" ")
    print("========================================")


    for k, test_controller in enumerate(test_controller_list):
        print("*********************")
        print("evaluate controller: {} ({}/{})".format(test_controller, k+1, len(test_controller_list)))
        ######################################################
        if test_controller == 'LFS':
            train_type = 'BP'
            model_type = 'DFNN'
        elif test_controller == 'PKD':
            train_type = 'PKD'
            model_type = 'DFNN'
        elif test_controller == 'PTM':
            model_type = 'analytical_model'
        else:
            raise Exception("controller type is not recognized")


        if test_controller == 'PTM':
            controller.load_gcc_model(model_type)
            controller.model.decode_json_file(join(load_PTM_param_path, "gc-"+MTM_ARM+"-"+SN +".json"))
        else:
            controller.load_gcc_model(model_type, load_model_path=load_model_path, use_net=use_net, train_type=train_type)

        time.sleep(0.2)



        # load test points from file
        q_mat = scipy.io.loadmat(join(save_testing_point_path, 'testing_points.mat'))['q_mat']
        ready_q_mat = scipy.io.loadmat(join(save_testing_point_path, 'testing_points.mat'))['ready_q_mat']
        sample_num = q_mat.shape[0]

        # create path to save results of experiemnts
        if not os.path.exists(save_result_path):
            os.makedirs(save_result_path)


        # initialize some variables for experiments
        if model_type == 'analytical_model':
            rate = 530
        else:
            rate = 370
        duration = 2.2
        controller.FIFO_buffer_size = rate * duration
        drift_pos_tensor = np.zeros((int(rate * duration), D, sample_num))
        drift_time = np.zeros((sample_num))
        drift_pos_cnt_arr = np.zeros((sample_num))
        drift_isExceedSafeVel_arr = np.full((sample_num), True, dtype=bool)
        sum_start_time = time.clock()

        for i in tqdm(range(sample_num)):
            loop_time = time.clock()
            controller.move_MTM_joint(ready_q_mat[i,:])
            time.sleep(0.3) # wait until arm stable
            controller.move_MTM_joint(q_mat[i,:])
            time.sleep(0.6) # wait until arm stable
            controller.clear_FIFO_buffer()
            print("   #### testing controller:({}) in config:({}/{})".format(test_controller, i+1, sample_num), end='')

            isExceedSafeVel = False
            gcc_time = time.clock()
            controller.isExceedSafeVel =False

            controller.start_gc()
            while not (controller.FIFO_pos_cnt==rate*duration or isExceedSafeVel): # break loop when finish duration or joint speed exceed safe limits
                time.sleep(0.001) # check the condition every 1ms
                isExceedSafeVel = controller.isExceedSafeVel

            # save experiment results
            gcc_time = time.clock() - gcc_time
            drift_pos_tensor[:,:,i] = controller.FIFO_pos
            drift_isExceedSafeVel_arr[i] = isExceedSafeVel
            drift_pos_cnt_arr[i] = controller.FIFO_pos_cnt
            drift_time[i] = gcc_time

            if verbose_silent_level<1:
                print("isExceedSafeVel: ", isExceedSafeVel)

            controller.stop_gc()
            controller.move_MTM_joint(controller.GC_init_pos_arr)
            time.sleep(0.1)

            # # old method to evaluate the process time
            # if verbose_silent_level <1:
            #     loop_time = time.clock() - loop_time
            #     sum_time = time.clock() - sum_start_time
            #     total_time = sum_time*(sample_num)/(i+1)
            #     print ("Time (GCC) is: ", gcc_time)
            #     print ("Time (loop time) is: ", loop_time)
            #     print("finish ("+str(i+1)+"/"+str(sample_num)+")"
            #           +" time:"+str(datetime.timedelta(seconds=sum_time))
            #           +" / "+str(datetime.timedelta(seconds=total_time)))

        # save experiement result to file
        if model_type == 'DFNN':
            file_name = use_net+'_'+train_type
        else:
            file_name = model_type
        scipy.io.savemat(join(save_result_path, file_name), {'drift_pos_tensor': drift_pos_tensor,
                                                             'drift_isExceedSafeVel_arr': drift_isExceedSafeVel_arr,
                                                             'drift_pos_cnt_arr': drift_pos_cnt_arr,
                                                             'drift_time':drift_time})

