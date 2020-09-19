# Deep Learning for Gravity Compensation using physical knowledge distillation

## Overview

* Codes for paper, *Learning Deep Nets for Gravitational Dynamics with Disturbance through Physical Knowledge Distillation*

## Features

- Data collection in Master Too Manipulator (MTM) of dVRK 
- Train deep neural network model for Gravity Compensation Controller (GCC) in MTM using pytorch
- GCC in MTM
- Evaluation experiments for GCC in MTM

## Requirement
* Ubuntu OS
* ROS
* python == 2.7
* Matlab (including robotics toolbox to run ROS API)

## Installation
1. Install [dVRK](https://github.com/jhu-cisst/cisst/wiki/Compiling-cisst-and-SAW-with-CMake#13-building-using-catkin-build-tools-for-ros).
2. Implement the [analytical solution](https://github.com/jhu-dvrk/dvrk-gravity-compensation) of GCC for dVRK (To obtain a Physical Teacher Model).
3. Install DFNN4GCC and install python packages
```sh
git clone https://github.com/linhongbin-ws/DFNN4GCC
cd DFNN4GCC
pip install -r requirements.txt
```

## Run
1. open a terminal
  ```sh
  roscore
  ```
  open another terminal and launch the dVRK console to control the MTM.
  ```sh
  qlacloserelays
  rosrun dvrk_robot dvrk_console_json -j <path-to-your-MTM-json-file>
  ```
<br /> 


2. open Matlab. Go to the "DFNN4GCC" directory. Type in the command line
  ```
  rosinit
  addpath('<path to /dvrk-ros>')
  ```
<br /> 

3. run `run_collect_train_data.m`, type following function with (1st argument `'MTML'` or `'MTMR'`, 2nd argument: 5digit of serial number, e.g. `'31519'`) in your matlab terminal according to your MTM,
for example: 
```Matlab
run_collect_train_data('MTMR', '31519')
```
<br /> 



After this, the program will collect training, validating, testing data for a MTM. It take around **4** hours to finish the process. There are 4 subprocesses running in serial, `wizard program`, `generating pivot points`, `Collision Checking`, `data collection`, `data pre-processing`.

* `wizard program` (required command inputs): A wizard program for setting the customized joint limits for specific dVRK system. This is important since it can identify the maximum joint ranges within a safety workspace. In anther words, it helps to improve the balance between safety and achieved performance. User need to type character to input some commands in the command dialog. There is a video to teaching user how to set joint limits using the Wizard Program [[video](https://www.youtube.com/watch?v=O8KM-scxTk4)].

* `generating pivot points`: Generate the pivot points representing the desired positions of a MTM for training, validating and testing data.

* `Collision Checking` (might require to press E-stop): Run through some pivot points to check if MTM will hit environment in the future data collection. Press E-stop if the MTM hit environment.

* `data collection`: Collecting data. It should take around 4+ hours. If you pass the `Collision Checking`, you no longer need to worry about that the MTM will hit environment during this 4 hours. You can do others work waiting the program.

* `data pre-processing`: Pre-processing the raw data to trigonometric representation.
<br /> 


4. copy the json file for the [analytical solution](https://github.com/jhu-dvrk/dvrk-gravity-compensation) to DFNN4GCC directory.
In the terminal

    ```sh
    cp /...path-to-json-file.../gc-MTMR-31519.json /...path-to-DFNN4GCC.../data/MTMR_31519/real
    ```
for example:

    ```sh
    cp /home/ben/gc-MTMR-31519.json /home/ben/DFNN4GCC/data/MTMR_31519/real
    ```
<br /> 


5. run `run_train.py` to train DFNN for Learn-from-Sratch(LfS) and Phyiscal-Knowledge-Distllation(PKD). Type in your terminal based on your MTM info, for example:
    ```sh
    chmod +x run_train.py
    python run_train.py --arm MTMR --sn 31519
    ```
<br /> 


6. run `run_Controller.py` to run your GCC. Type in your terminal based on your MTM info and the controller(`LFS`, `PTM`, `PTM`) you want to evaluate, for example:
    ```sh
    chmod +x run_controller.py
    python run_controller.py  --arm MTMR --sn 31519 --controller PTM
    ```
    Type `Ctrl+C` to stop safely.


## Experiment Evaluation (Optional)
This part of code is to reproduce experiments (Trajectory Test and Drift Test) in our paper. 

### 1.Trajectory Test (offline test)
run in matlab terminal based your MTM info and sampling points number for experiment, for example
```Matlab
run_collect_traj_test_data('MTMR', '31519', 400)
```

To reproduce the figure in our paper, you can type in terminal based on your MTM info, for example
```
python run_plot_trajectoryTest_D6_SinCosInput.py --arm MTMR --sn 31519 --sample_num 200
```
and it will generate the following figure based your testing points. (Figure example: [link](https://github.com/linhongbin-ws/DFNN4GCC/blob/controller-evaluation/data/MTMR_28002/real/dirftTest/N4/D6_SinCosInput/dual/result/TrajTest_AbsRMS.pdf))

(*For Trajectory test, the program will collect data in random configurations and save them to mat files for offline evaluation. For each testing point, the program will first move MTM to a ready configuration and then move to desired configuration. The reason for this is to test how the controller react to direction-depedent disturbance mainly causing by hystestic effect from cables.*)
<br /> 


### 2.Drift Test (online test)
run in terminal based your MTM info and sampling points for experiments, for example
```sh
python run_collect_drift_test_data.py --arm MTMR --sn 31519 --sample_num 200
```

To reproduce the figure in our paper, you can type in terminal based on your MTM info, for example
```
python run_plot_driftTest_D6_SinCosInput.py --arm MTMR --sn 31519
```
and it will generate the following figure based your testing points. (Figure example: [link](https://github.com/linhongbin-ws/DFNN4GCC/blob/controller-evaluation/data/MTMR_31519/real/dirftTest/N4/D6_SinCosInput/dual/result/DriftTest_all.pdf))

*(For drift test, it will take one hour for each controller and the program will test 3 controller automatically for 3 hour in total. For each testing point, the program will first move to a ready configuration and then move to desired configuration. The reason is the same with the one of Trajectory Test. Afteward the controller will switch from PID to feedfoward GCC for about 2 seconds. Joint drifts are recorded. Then the program will switch back to PID controller to move to next testing point. In some extreme cases that makes the MTM move too fast under evaluated GCC, the program will switch back to PID if the joint velocity exceed limits to prevent damages to a MTM.)*

If you stop the program during the process and want to restart it and only want to collect a specific controller without going through re-collecting 3 controllers. You can go to python script `run_Plot_DriftTest_D6_SinCosInput.py` and uncomment and change the following line
```python
    test_controller_list = ['PTM', 'LFS', 'PKD']
    #test_controller_list = ['PTM'] # uncomment this line if you only want to collect dirft test data for only one controller
```

## Data Sharing
Since our program can be reproduced in MTMs for a general dVRK system, it is encourage to sharing your data to dVRK community for further research analysis (such as study on transfer learning to further reduce the required amount of training data.)

To push the data for your dVRK system to our github repository, just type in terminal
```sh
git add ./data/your-system-folder/*
git commit -m `add <MTM Name>_<Serial Number>`
git push
```

