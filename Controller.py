from __future__ import print_function
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import dvrk
from loadModel import get_model, load_model
from AnalyticalModel import *
import json
import time
import os




class Controller():

    # default controller settings
    safe_upper_torque_limit_arr = np.array([0.2, 0.8, 0.6, 0.2, 0.2, 0.2, 0])    # saturate upper bound of output torques
    safe_lower_torque_limit_arr = np.array([-0.2, -0.1, 0, -0.3, -0.1, -0.1, 0]) # saturate lower bound of output torques
    db_vel_arr = np.array([0.02, 0.02, 0.02, 0.01, 0.008, 0.008, 0.01])          # dead band for velocity coefficients
    sat_vel_arr = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])                  # saturated bound for velocity coefficients
    fric_comp_ratio_arr = np.array([0.7, 0.01, 0.5, 0.4, 0.2, 0.2, 1])           # compensated ratio for friction compensation
    GC_init_pos_arr = np.radians(np.array([0, 0, 0, 0, 90, 0, 0]))               # move MTM to initial pose when starting controller
    jnt_upper_limit = np.radians(np.array([40, 45, 34, 190, 175, 40]))           # default upper joint limit
    jnt_lower_limit = np.radians(np.array([-40, -14, -34, -80, -85, -40]))       # default lower joint limit
    jnt_coup_upper_limit = np.radians(41)                                        # default upper couple joint limit
    jnt_coup_lower_limit = np.radians(-11)                                       # default lower couple joint limit
    jnt_coup_limit_index = [1,2]                                                 # joint 2 and Joint 3 as couple limit joints
    ready_q_margin = np.radians(np.array([5,5,5,5,5,5]))                         # default joint margin for ready pose
    jnt_limit_check_margin =  np.radians(0.1)
    safe_vel_limit = np.array([6,6,6,6,6,6,100])                                 # stop controller when joint velocity exceed such a velocity range
    D = 6                                                                        # control Joint 1 to 6
    device = 'cpu'                                                               # prediction use cpu in pytorch
    verbose_silent_level = 0                                                     # 0 for print everything, 1 for print something, 2 for print nothing

    # variables
    count = 0
    model = None
    isExceedSafeVel = False
    isOutputGCC = False
    isGCCRuning = False
    FIFO_buffer_size = 1000
    FIFO_pos = np.zeros((FIFO_buffer_size,D))
    FIFO_pos_cnt = 0
    current_pos_lst = None

    def __init__(self, MTM_ARM):
        # define ros node
        rospy.init_node(MTM_ARM + 'GCC_controller', anonymous=True)
        self.MTM_ARM = MTM_ARM

        # define topics
        self.pub_tor_topic = '/dvrk/' + MTM_ARM + '/set_effort_joint'
        self.sub_pos_topic = '/dvrk/' + MTM_ARM + '/state_joint_current'
        self.pub_isDefaultGCC_topic = '/dvrk/' + MTM_ARM + '/set_gravity_compensation'

        # define publisher
        self.pub_tor = rospy.Publisher(self.pub_tor_topic, JointState, latch = True, queue_size = 10)
        self.pub_isDefaultGCC = rospy.Publisher(self.pub_isDefaultGCC_topic, Bool, latch = True, queue_size = 1)




        # shut down dvrk default GCC
        self.set_default_GCC_mode(False)

        # define dvrk python api
        self.mtm_arm = dvrk.mtm(MTM_ARM)

        # keyboard shutdown function
        rospy.on_shutdown(self.shutdown)

        # wait for ros communication
        time.sleep(0.1)




    def load_gcc_model(self, model_type, load_model_path=None, use_net=None, train_type=None):
        if model_type == 'analytical_model':
            self.model =  MTM_MLSE4POL()
        elif model_type == 'DFNN':
            self.model = get_model('MTM', use_net, self.D, device='cpu')
            self.model, _, _ = load_model(load_model_path, use_net + '_' + train_type,  self.model)
        else:
            raise Exception("model type is not support.")

    def start_gc(self):

        # check if model is assigned
        self.isExceedSafeVel = False

        if self.model is None:
            print("you should load the model before call start_gc().")
            return 0

        self.sub_pos = rospy.Subscriber(self.sub_pos_topic, JointState, self.sub_pos_cb_with_gcc)
        self.set_isOutputGCC(True)
        self.isGCCRuning = True
        if self.verbose_silent_level == 0:
            print("GCC start")


    def stop_gc(self):
        self.sub_pos.unregister()
        self.mtm_arm.move_joint(self.GC_init_pos_arr)
        self.isGCCRuning = False
        if self.verbose_silent_level == 0:
            print("GCC stop...")


    def shutdown(self):
        try:
            self.stop_gc()
        except:
            print("ROS losses connection")

        self.isGCCRuning = False
        print("GCC Shutdown...")

    def update_isExceedSafeVel(self, vel_arr):
        abs_vel_arr = np.abs(vel_arr)
        for i in range(self.D):
            if abs_vel_arr[i] > self.safe_vel_limit[i]:
                self.isExceedSafeVel = True
                break
            else:
                self.isExceedSafeVel = False


    def pub_zero_torques(self):
        msg = JointState()
        msg.effort = np.zeros(7).tolist()
        self.pub_tor.publish(msg)
        time.sleep(0.4)


    def dbs_vel(self, joint_vel, bd_vel, sat_vel, fric_comp_ratio):
        """
        Deadband segmented function
        :param bd_vel: dead band velocity
        :param sat_vel: saturated velocity
        :param fric_comp_ratio: compensate ratio
        :return: direction coeifficent 0~1
        """
        if joint_vel >= sat_vel:
            sign_vel = 0.5 + 0.5 * fric_comp_ratio;
        elif joint_vel <= -sat_vel:
            sign_vel = 0.5 - 0.5 * fric_comp_ratio;
        elif joint_vel <= bd_vel and joint_vel >= -bd_vel:
            sign_vel = 0.5
        elif joint_vel > bd_vel and joint_vel < sat_vel:
            sign_vel = 0.5 * fric_comp_ratio * (joint_vel - bd_vel) / (sat_vel - bd_vel) + 0.5
        elif joint_vel < -bd_vel and joint_vel > -sat_vel:
            sign_vel = -0.5 * fric_comp_ratio * (-joint_vel - bd_vel) / (sat_vel - bd_vel) + 0.5
        else:
            raise Exception("joint_vel is not in if range")

        return sign_vel


    def sub_pos_cb_with_gcc(self, data):


        self.current_pos_lst = data.position
        pos_lst = data.position[:-1]  # select 1-6 joints
        vel_lst = data.velocity[:-1]
        effort_lst = data.effort[:-1]

        pos_arr = np.array(pos_lst)
        vel_arr = np.array(vel_lst)
        effort_arr = np.array(effort_lst)

        tor_arr = self.predict(pos_arr, vel_arr)
        tor_arr = self.bound_tor(tor_arr)




        if self.isOutputGCC:
            msg = JointState()
            output_lst = tor_arr.tolist()
            output_lst.append(0.0)
            msg.effort = output_lst
            self.pub_tor.publish(msg)
        else:
            msg = JointState()
            msg.effort = np.zeros(7).tolist()
            self.pub_tor.publish(msg)

        self.update_isExceedSafeVel(vel_arr)
        self.update_FIFO_buffer(pos_arr)

    # model predict function
    def clear_FIFO_buffer(self):
        self.FIFO_pos = np.zeros((int(self.FIFO_buffer_size), self.D))
        self.FIFO_pos_cnt = 0

    def update_FIFO_buffer(self, pos_arr):
        tmp = self.FIFO_pos[:-1,:]
        tmp = np.concatenate((pos_arr.reshape(1,-1), tmp), axis=0)
        self.FIFO_pos = tmp
        self.FIFO_pos_cnt += 1
        if self.FIFO_pos_cnt > self.FIFO_buffer_size:
            self.FIFO_pos_cnt = self.FIFO_buffer_size

    def predict(self, pos_arr, vel_arr):
        """
        :param SinCos_pos_arr: [sin(q), cos(q)]
        :param vel_arr: velocity array
        :return: tor_arr
        """

        if self.model is not None:
            pos = pos_arr.reshape(1,-1)
            SinCos_pos = np.concatenate((np.sin(pos), np.cos(pos)), axis=1)
            tor_pos = self.model.predict_NP(np.concatenate((SinCos_pos, np.ones((1, self.D))), axis=1))
            tor_neg = self.model.predict_NP(np.concatenate((SinCos_pos, np.zeros((1, self.D))), axis=1))

            sign_vel_vec = np.zeros((1, 6))
            for i in range(6):
                sign_vel_vec[0][i] = self.dbs_vel(vel_arr[i], self.db_vel_arr[i], self.sat_vel_arr[i], self.fric_comp_ratio_arr[i])

            tor = np.multiply(tor_pos, sign_vel_vec) + np.multiply(tor_neg, 1 - sign_vel_vec)
            tor_arr = tor[0]
        else:
            tor_arr = np.zeros((self.D))
        return tor_arr

    # saturate the output torques
    def bound_tor(self, tor_arr):
        tor = tor_arr
        for i in range(self.D):
            if tor[i] >= self.safe_upper_torque_limit_arr[i]:
                tor[i] = self.safe_upper_torque_limit_arr[i]
            elif tor[i] <= self.safe_lower_torque_limit_arr[i]:
                tor[i] = self.safe_lower_torque_limit_arr[i]
        return tor

    def set_current_pos(self):
        self.move_MTM_joint(np.array(self.mtm_arm.get_current_joint_position()),
                            interpolate=True, blocking=True)


    # publish topic: set_gravity_compensation
    def set_default_GCC_mode(self, is_enable):
        if is_enable:
            self.pub_isDefaultGCC.publish(Bool(data=True))
        else:
            self.pub_isDefaultGCC.publish(Bool(data=False))
        time.sleep(0.4)



    def ros_spin(self):
        while not rospy.is_shutdown():
            pass
        self.stop_gc()


    def set_isOutputGCC(self, isOutputGCC):
        self.isOutputGCC = isOutputGCC

    def move_MTM_joint(self, jnt_pos_arr, interpolate = True, blocking = True):


        self.mtm_arm.move_joint(jnt_pos_arr, interpolate = interpolate, blocking = blocking)
        if self.verbose_silent_level == 0:
            print("Moving MTM to desired jnt position: " + ', '.join("{:.1f}".format(ang) for ang in np.degrees(jnt_pos_arr).tolist()))
        # time.sleep(0.5)

    def random_testing_configuration(self, sample_num):
        count = 0
        q_mat = np.zeros((sample_num, self.D))
        ready_q_mat = np.zeros((sample_num, self.D))

        while not count==sample_num:
                rand_arr = np.random.rand(self.D)
                q_arr = rand_arr * (self.jnt_upper_limit - self.jnt_lower_limit) + self.jnt_lower_limit

                u_arr = np.zeros((self.D))
                rand_arr = np.random.rand(self.D)
                for j in range(self.D):
                    u_arr[j] = 1 if rand_arr[j]>0.5 else 0

                dir_arr = u_arr - (1-u_arr)

                ready_q_arr = q_arr - np.multiply(self.ready_q_margin, dir_arr)

                if self.is_within_joint_limit(ready_q_arr, self.jnt_limit_check_margin) and self.is_within_joint_limit(q_arr, self.jnt_limit_check_margin):
                    q_mat[count, :] = q_arr
                    ready_q_mat[count, :] = ready_q_arr
                    count = count + 1

        q_mat = np.concatenate((q_mat, np.zeros((q_mat.shape[0], 1))), axis=1)
        ready_q_mat = np.concatenate((ready_q_mat, np.zeros((ready_q_mat.shape[0], 1))), axis=1)

        return q_mat, ready_q_mat




    def is_within_joint_limit(self, q_arr, limit_margin):
        is_within_lower_limit = (self.jnt_lower_limit+limit_margin)<=q_arr
        is_within_lower_limit = all(is_within_lower_limit.tolist())

        is_within_upper_limit = (self.jnt_upper_limit-limit_margin)>=q_arr
        is_within_upper_limit = all(is_within_upper_limit.tolist())

        is_within_coup_lower_limit = (self.jnt_coup_lower_limit+limit_margin) <= q_arr[self.jnt_coup_limit_index[0]] + q_arr[self.jnt_coup_limit_index[1]]
        is_within_coup_upper_limit = (self.jnt_coup_upper_limit-limit_margin) >= q_arr[self.jnt_coup_limit_index[0]] + q_arr[self.jnt_coup_limit_index[1]]

        return all([is_within_lower_limit, is_within_upper_limit, is_within_coup_lower_limit, is_within_coup_upper_limit])

    def load_jointLimit_json(self, json_file_str):
        if not os.path.isfile(json_file_str):
            print("cannot find {}".format(json_file_str))
            return False

        print('loading file {}'.format(json_file_str))
        with open(json_file_str) as json_file:
            data = json.load(json_file)
            self.jnt_upper_limit = np.radians(data['joint_pos_upper_limit'])
            self.jnt_lower_limit = np.radians(data['joint_pos_lower_limit'])
            self.jnt_coup_upper_limit = np.radians(data['coupling_upper_limit'])
            self.jnt_coup_lower_limit = np.radians(data['coupling_lower_limit'])

        print('Updating joint limits:',end='\n')
        print('jnt_upper_limit:', end='')
        print(', '.join('{}'.format(k) for k in data['joint_pos_upper_limit']))
        print('jnt_lower_limit:', end='')
        print(', '.join('{}'.format(k) for k in data['joint_pos_lower_limit']))
        print('coupling_upper_limit:', end='')
        print(data['coupling_upper_limit'])
        print('coupling_lower_limit:', end='')
        print(data['coupling_lower_limit'])

        return True

# # #
# # # # # #
# # # # # # # #
# # # # # # # #
# # # # # # # # test controller function
# MTM_ARM = 'MTMR'
# use_net = 'ReLU_Dual_UDirection'
# load_model_path = join("data", "MTMR_28002", "real", "uniform", "N4", 'D6_SinCosInput', "dual", "result", "model")
# train_type = 'PKD'
# # model_type = 'DFNN'analytical_model
# model_type = 'analytical_model'
#
#
# controller = Controller(MTM_ARM)
# controller.load_gcc_model(model_type, load_model_path=load_model_path, use_net=use_net, train_type=train_type)
# # controller.load_gcc_model(model_type)
# # pdb.set_trace()
# time.sleep(1)
# controller.move_MTM_joint(controller.GC_init_pos_arr)
# time.sleep(4)
# controller.start_gc()
# time.sleep(4)
# # controller.stop_gc()
# controller.ros_spin()
# # controller.stop_gc()
# # controller.move_MTM_joint(controller.GC_init_pos_arr)
# # time.sleep(4)
# # #
