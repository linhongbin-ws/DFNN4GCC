from __future__ import print_function
from os.path import join
import time
from Controller import Controller
import argparse
import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', type=str, required=True, help="MTML or MTMR")
    parser.add_argument('--sn', type=str, required=True, help="Serial Number: e.g. 28002")
    parser.add_argument('--controller', type=str, required=True, help="controller type: PTM, LFS or PKD,  PTM(Physical Teacher Model), LFS(Learn From Scratch), PKD(Physical Knowledge Distillation)")
    args = parser.parse_args()
    MTM_ARM = args.arm
    SN = args.sn
    controller_type = args.controller

    if not (MTM_ARM == 'MTML' or MTM_ARM == 'MTMR'):
        print('The first argument(arm)  should be either MTML or MTMR')
        sys.exit()
    if not (len(SN)==5):
        print('The second argument(sn) should be a string with five number, e.g.: 28008')
        sys.exit()
    if not controller_type in ['LFS', 'PKD', 'PTM']:
        print('controller_type : {} is not one of [LFS, PKD, PTM]'.format(controller_type))
        sys.exit()
    #####################################################################


    use_net = 'ReLU_Dual_UDirection'
    load_model_path = join("data", MTM_ARM+'_'+SN, "real", "uniform", "N4", 'D6_SinCosInput', "dual", "result", "model")
    load_PTM_param_path = join("data",  MTM_ARM+'_'+SN, "real", "gc-"+MTM_ARM+"-"+SN +".json")



    ######################################################
    if controller_type == 'LFS':
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
