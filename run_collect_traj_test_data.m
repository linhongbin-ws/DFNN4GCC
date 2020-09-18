function run_collect_traj_test_data(ARM_NAME, SN, N_traj_test)
    % overview: collect pivot point data from MTM for testing data in offline trajectory test to evaluate controllers
    % arguments:
    %   ARM_NAME: 'MTML' or 'MTMR' (ARM Type)
    %   SN: '12345' (Serial Number)
    % output files:
    %       <....>/random/<....> (testing data)
    %%%%%%%%%%%%% Generate matrirx for pivot points %%%%%%%%%%%%
    
    dataCollection_config_customized_str = fullfile('data', [ARM_NAME,'_', SN], 'real','dataCollection_config_customized.json')
    if ~isfile(dataCollection_config_customized_str)
        fprintf("cannot find %s, you might need to run run_collect_train_data.m first.")
        return
    end
    
    setup_matlab;
    
    % generate pivot points for validating data
    config_mat = generate_config_pivot_points_random(dataCollection_config_customized_str, N_traj_test);
    pivot_points_path_validate = fullfile('data', [ARM_NAME,'_',SN], 'real', 'random', ['N', int2str(N_traj_test)] ,'raw_data');
    if ~exist(pivot_points_path_validate, 'dir')
       mkdir(pivot_points_path_validate);
    end
    fprintf('saving pivot points for Trajectory Test data..\n')
    save(fullfile(pivot_points_path_validate, 'desired_pivot_points.mat'), 'config_mat' ,'N_traj_test');


    %%%%%%%%%%%%%%%%% Collect data %%%%%%%%%%%%%%%%
    %%%%
    % collect training data, about 4 hour.
    %%%%

    % collect validating data, a couple minutes
    is_reverse = false;
    [current_position, desired_effort] = collect_data(ARM_NAME,...
                                fullfile(pivot_points_path_validate, 'desired_pivot_points.mat'), is_reverse); 
    save_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'random', ['N',int2str(N_traj_test)],'raw_data');
    save(fullfile(save_path, 'joint_pos'),'current_position');
    save(fullfile(save_path, 'joint_tor'),'desired_effort');


    %%%%%%%%%%%%%%%%%%%%% Data Pre-processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % data processsing

    root_path =  fullfile('data', [ARM_NAME, '_',SN], 'real', 'random', ['N',int2str(N_traj_test)]);
    is_dual = false;
    rawdataProcess(root_path, is_dual);
    rosshutdown;
    fprintf('run_collect_traj_test_data.m program have finished!!\n');
end
