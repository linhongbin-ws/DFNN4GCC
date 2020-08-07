import rospy
from sensor_msgs.msg import JointState

from std_msgs.msg import UInt8MultiArray

from std_msgs.msg import Bool
import dvrk
import numpy as np
from os.path import join
import torch
from Net import *
from loadModel import get_model, load_model
import time

from AnalyticalModel import *
import pdb
from Controller import Controller

#####################################
# settings
MTM_ARM = 'MTMR'
SN = '31519'


# controller_type = 'LfS' # Learn-from-Sratch approach
#controller_type = 'PKD' # Physical Knowledge Distillation
controller_type = 'PTM' # Physical Teacher Model



use_net = 'ReLU_Dual_UDirection'
load_model_path = join("data", MTM_ARM+'_'+SN, "real", "uniform", "N4", 'D6_SinCosInput', "dual", "result", "model")
load_PTM_param_path = join("data",  MTM_ARM+'_'+SN, "real", "gc-"+MTM_ARM+"-"+SN +".json")



######################################################
if controller_type == 'LfS':
    train_type = 'BP'
    model_type = 'DFNN'
elif controller_type == 'PKD':
    train_type = 'PKD'
    model_type = 'DFNN'
elif controller_type == 'PTM':
    model_type = 'analytical_model'
else:
    raise Exception("controller type is not recognized")


controller = Controller(MTM_ARM)
if controller_type == 'PTM':
    controller.load_gcc_model(model_type)
    controller.model.decode_json_file(load_PTM_param_path)
else:
    controller.load_gcc_model(model_type, load_model_path=load_model_path, use_net=use_net, train_type=train_type)
# controller.load_gcc_model(model_type)
# pdb.set_trace()
time.sleep(1)
controller.move_MTM_joint(controller.GC_init_pos_arr)
time.sleep(4)
controller.start_gc()
#time.sleep(4)
# controller.stop_gc()
print("Press Ctrl+C to Stop")
controller.ros_spin()

# controller.stop_gc()
# controller.move_MTM_joint(controller.GC_init_pos_arr)
# time.sleep(4)
# #
