function safeCollisionCheck(config_mat_safeCheck, ARM_NAME)
    %  Author(s):  Hongbin LIN, Samuel Au
    %  comments: run through testing pivot point of collision checking to ensure safety
    
%     ARM_NAME = 'MTMR'
%     SN = '31519'
%     load_file = fullfile(desired_pivot_points_str)
    mtm_arm = mtm(ARM_NAME);

    config_mat_safeCheck = config_mat_safeCheck(:, [2:2:end]); % skip checking Joint 6 for saving time
    for i = 1:size(config_mat_safeCheck, 2)
        pos = [config_mat_safeCheck(:,i).', 0];
        mtm_arm.move_joint(deg2rad(pos));

        disp(sprintf("finish: (%d / %d), press E-stop if MTM hit environment", i, size(config_mat_safeCheck, 2)))
    end
end