import time
import scipy.io
import datetime

from Controller import Controller
import numpy as np
from os.path import join
import scipy.io
import os


MTM_ARM = 'MTMR'
save_testing_point_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual")

sample_num = 200
D = 6

controller = Controller(MTM_ARM)
q_mat, ready_q_mat = controller.random_testing_configuration(sample_num)




if not os.path.exists(save_testing_point_path):
    os.makedirs(save_testing_point_path)



file_name = "testing_points.mat"
scipy.io.savemat(join(save_testing_point_path, file_name), {'q_mat': q_mat,
                             'ready_q_mat': ready_q_mat})


MTM_ARM = 'MTMR'
use_net = 'ReLU_Dual_UDirection'
load_model_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual", "result", "model")
load_testing_point_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual")
save_result_path = join("data", "MTMR_28002", "real", "dirftTest", "N4", 'D6_SinCosInput', "dual", "result")
train_type = 'BP'
# model_type = 'analytical_model'
model_type = 'DFNN'


D = 6

controller = Controller(MTM_ARM)
controller.load_gcc_model(model_type, load_model_path=load_model_path, use_net=use_net, train_type=train_type)
time.sleep(1)

q_mat = scipy.io.loadmat(join(load_testing_point_path, 'testing_points.mat'))['q_mat']
ready_q_mat = scipy.io.loadmat(join(load_testing_point_path, 'testing_points.mat'))['ready_q_mat']

sample_num = q_mat.shape[0]

# for j in range(6):
#     print("---------- ")
#     print("jnt_upper_limit: ", np.degrees(controller.jnt_upper_limit[j]))
#     print("jnt_lower_limit: ", np.degrees(controller.jnt_lower_limit[j]))
#     for i in range(7):
#         print("q_mat :", np.degrees(q_mat[i, j]))
#         print("ready_q_mat :",np.degrees(ready_q_mat[i, j]))
#
# print("++++++++++++")
# print("jnt_coup_upper_limit: ", np.degrees(controller.jnt_coup_upper_limit))
# print("jnt_coup_lower_limit: ", np.degrees(controller.jnt_coup_lower_limit))
# for i in range(7):
#     print("q_mat :", np.degrees(q_mat[i, 1]+q_mat[i, 2]))
#     print("ready_q_mat :", np.degrees(ready_q_mat[i, 1]+ready_q_mat[i, 2]))
#pdb.set_trace()


if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)


# pdb.set_trace()
if model_type == 'analytical_model':
    rate = 530
else:
    rate = 370
duration = 2.2
controller.FIFO_buffer_size = rate * duration

drift_pos_tensor = np.zeros((rate * duration, D, sample_num))
drift_time = np.zeros((sample_num))
drift_pos_cnt_arr = np.zeros((sample_num))
drift_isExceedSafeVel_arr = np.full((sample_num), True, dtype=bool)

sum_start_time = time.clock()

for i in range(sample_num):
    loop_time = time.clock()
    controller.move_MTM_joint(ready_q_mat[i,:])
    time.sleep(0.3)
    controller.move_MTM_joint(q_mat[i,:])
    time.sleep(1)

    controller.clear_FIFO_buffer()
    controller.start_gc()

    isExceedSafeVel = False
    gcc_time = time.clock()
    controller.isExceedSafeVel =False

    while not (controller.FIFO_pos_cnt==rate*duration or isExceedSafeVel):
        # print(controller.FIFO_pos_cnt)
        time.sleep(0.001)
        isExceedSafeVel = controller.isExceedSafeVel

    drift_pos_tensor[:,:,i] = controller.FIFO_pos
    drift_isExceedSafeVel_arr[i] = isExceedSafeVel
    drift_pos_cnt_arr[i] = controller.FIFO_pos_cnt

    gcc_time = time.clock() - gcc_time
    drift_time[i] = gcc_time


    print("isExceedSafeVel: ", isExceedSafeVel)
    # print("Buffer: ", controller.FIFO_pos)
    # print("Buffer count ", controller.FIFO_pos_cnt)


    controller.stop_gc()
    controller.move_MTM_joint(controller.GC_init_pos_arr)
    time.sleep(0.2)

    loop_time = time.clock() - loop_time
    sum_time = time.clock() - sum_start_time
    total_time = sum_time*(sample_num)/(i+1)
    print ("Time (GCC) is: ", gcc_time)
    print ("Time (loop time) is: ", loop_time)
    print("finish ("+str(i+1)+"/"+str(sample_num)+")"
          +" time:"+str(datetime.timedelta(seconds=sum_time))
          +" / "+str(datetime.timedelta(seconds=total_time)))


if model_type == 'DFNN':
    file_name = use_net+'_'+train_type
else:
    file_name = model_type
scipy.io.savemat(join(save_result_path, file_name), {'drift_pos_tensor': drift_pos_tensor,
                             'drift_isExceedSafeVel_arr': drift_isExceedSafeVel_arr,
                             'drift_pos_cnt_arr': drift_pos_cnt_arr,
                             'drift_time':drift_time})

