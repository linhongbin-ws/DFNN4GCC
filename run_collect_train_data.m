function run_collect_train_data(ARM_NAME, SN)
    % overview: collect pivot point data from MTM for training nerual network controllers
    % arguments:
    %   ARM_NAME: 'MTML' or 'MTMR' (ARM Type)
    %   SN: '12345' (Serial Number)
    % output files:
    %       <....>/dataCollection_config_customized.json (params for customized joint limits of a MTM)
    %       <....>/uniform/<....> (training data)
    %       <....>/random/<....> (testing data)
    
    %%%%%%%%%%%%% Hyperparam %%%%%%%%%%%%%%
    N_train = 4; % N is the sampling number for each joint, N=4 for our experiment
    N_validate = 160; % param for random sampling, 160 randomly sampled points    
    N_test = 40; % param for random sampling, 40 randomly sampled points
    
    %%%%%%%%%%%%% Wizard program for indentifying joint limits for daVici System %%%%%%%%%%%%%%%%%%%%%%

    fprintf('running run_collectDatat program..\n')
    fprintf('ARM_NAME: %s\n', ARM_NAME)
    fprintf('SN: %s\n', SN)
    % A wizard program to identify the custom joint limits (to ensure MTM to move within your specific workspace)
    dataCollection_config_customized_str = wizard4JntLimit(ARM_NAME, SN);

    %%%%%%%%%%%%% Generate matrirx for pivot points %%%%%%%%%%%%

    % generate pivot points for training data
    [config_mat, config_mat_safeCheck] = generate_config_pivot_points_with_same_interval(dataCollection_config_customized_str, N_train);
    pivot_points_path_train = fullfile('data', [ARM_NAME,'_',SN], 'real', 'uniform', ['N', int2str(N_train)],'raw_data');
    if ~exist(pivot_points_path_train, 'dir')
       mkdir(pivot_points_path_train);
    end
    fprintf('saving pivot points for traing data..\n')
    save(fullfile(pivot_points_path_train, 'desired_pivot_points.mat'), 'config_mat' ,'N_train');

    % generate pivot points for validating data
    config_mat = generate_config_pivot_points_random(dataCollection_config_customized_str, N_validate);
    pivot_points_path_validate = fullfile('data', [ARM_NAME,'_',SN], 'real', 'random', ['N', int2str(N_validate)] ,'raw_data');
    if ~exist(pivot_points_path_validate, 'dir')
       mkdir(pivot_points_path_validate);
    end
    fprintf('saving pivot points for validating data..\n')
    save(fullfile(pivot_points_path_validate, 'desired_pivot_points.mat'), 'config_mat' ,'N_validate');

    % generate pivot points for testing data
    config_mat = generate_config_pivot_points_random(dataCollection_config_customized_str, N_test);
    pivot_points_path_test = fullfile('data', [ARM_NAME,'_',SN], 'real', 'random', ['N', int2str(N_test)] ,'raw_data');
    if ~exist(pivot_points_path_test, 'dir')
       mkdir(pivot_points_path_test);
    end
    fprintf('saving pivot points for testing data..\n')
    save(fullfile(pivot_points_path_test, 'desired_pivot_points.mat'), 'config_mat' ,'N_test');

    % collision checking program ( to ensure MTM not to hit environment for training data only)
    fprintf('running collision checking program..\n')
    safeCollisionCheck(config_mat_safeCheck, ARM_NAME);


    %%%%%%%%%%%%%%%%% Collect data%%%%%%%%%%%%%%%%
    %%%%
    % collect training data, about 4 hour.
    %%%%

    % in non-reverse order, about 2 hour
    is_reverse = false;
    [current_position, desired_effort] = collect_data(ARM_NAME,...
                                fullfile(pivot_points_path_train, 'desired_pivot_points.mat'), is_reverse); % 2 hour
    save_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', ['N',int2str(N_train)],'raw_data');
    save(fullfile(save_path, 'joint_pos'),'current_position');
    save(fullfile(save_path, 'joint_tor'),'desired_effort');

    % in reverse order, about 2 hour
    is_reverse = true;
    [current_position, desired_effort] = collect_data(ARM_NAME,...
                                fullfile(pivot_points_path_train, 'desired_pivot_points.mat'), is_reverse); % 2 hour
    save_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', ['N',int2str(N_train)],'raw_data');
    save(fullfile(save_path, 'joint_pos_reverse'),'current_position');
    save(fullfile(save_path, 'joint_tor_reverse'),'desired_effort');

    % collect validating data, a couple minutes
    is_reverse = false;
    [current_position, desired_effort] = collect_data(ARM_NAME,...
                                fullfile(pivot_points_path_validate, 'desired_pivot_points.mat'), is_reverse); 
    save_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'random', ['N',int2str(N_validate)],'raw_data');
    save(fullfile(save_path, 'joint_pos'),'current_position');
    save(fullfile(save_path, 'joint_tor'),'desired_effort');


    % collect testing data, a couple minutes
    is_reverse = false;
    [current_position, desired_effort] = collect_data(ARM_NAME,...
                                fullfile(pivot_points_path_test, 'desired_pivot_points.mat'), is_reverse); 
    save_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'random', ['N',int2str(N_test)],'raw_data');
    save(fullfile(save_path, 'joint_pos'),'current_position');
    save(fullfile(save_path, 'joint_tor'),'desired_effort');


    %%%%%%%%%%%%%%%%%%%%% Data Pre-processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % data processsing
    root_path = fullfile('data', [ARM_NAME, '_',SN], 'real', 'uniform', ['N',int2str(N_train)]);
    is_dual = true;
    rawdataProcess(root_path, is_dual);

    root_path =  fullfile('data', [ARM_NAME, '_',SN], 'real', 'random', ['N',int2str(N_validate)]);
    is_dual = false;
    rawdataProcess(root_path, is_dual);

    root_path =  fullfile('data', [ARM_NAME, '_',SN], 'real', 'random', ['N',int2str(N_test)]);
    is_dual = false;
    rawdataProcess(root_path, is_dual);
end

