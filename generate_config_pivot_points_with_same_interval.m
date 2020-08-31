function [config_mat, config_mat_safeCheck] = generate_config_pivot_points_with_same_interval(dataCollection_config_customized_str, N)
    %  Author(s):  Hongbin LIN, Samuel Au
    %  comments: generate the pivot points, which represent the desired positions for data collection,
    %            using systematic sampling within joint limits of MTM
   
    
    
    % ARM_NAME = 'MTMR'
    % SN = '31519'
%     root_path = fullfile('data', [ARM_NAME, '_',SN], 'real')

    fid = fopen(dataCollection_config_customized_str);
    if fid<3
        error('cannot open file dataCollection_config_customized.json, please check the path');
    end
    raw = fread(fid, inf);
    str = char(raw');
    config = jsondecode(str);
    fclose(fid);

    joint_pos_upper_limit = config.joint_pos_upper_limit.'; 
    joint_pos_lower_limit = config.joint_pos_lower_limit.'; 
    coupling_index_list = {config.coupling_index_list.'};
    coupling_upper_limit = [config.coupling_upper_limit]; 
    coupling_lower_limit = [config.coupling_lower_limit];

    joint_pivot_num_list = N * [1,1,1,1,1,1];

    % margin to the joint limit, e.g.: joint limit:-180~180, joint_lim_margin_list = [0.5, ....], data collection:-180+0.5~180-0.5
    joint_lim_margin_list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

    pivot_mat = [];
    joint3_coup_pivot_mat = [];
    for i = 1:size(joint_pos_upper_limit,2)
        % treat joint 3 as special case dealing with coupling limits
        if(i==3)
            pivot_mat = [pivot_mat,zeros(joint_pivot_num_list(3),1)]; % asign zero to not store information
        else
            l_limit = joint_pos_lower_limit(i)+joint_lim_margin_list(i);
            u_limit = joint_pos_upper_limit(i)-joint_lim_margin_list(i);
            if l_limit >= u_limit
                error(sprintf('Joint limit %d setting error: l_limit: %d should be smaller than u_limit: %d\n', i, l_limit, u_limit))
            end
            pivot_mat = [pivot_mat,...
                                  linspace(l_limit, u_limit, joint_pivot_num_list(i))'];
                     
        end
    end
    
    
    % a lookup table for generating joint 3 pivot points based on joint 2 pivot points 
    %             joint2 p1, p2, p3, p4
    %    joint3
    %         p1         x  x    x   x
    %         p2         x  x    x   x 
    %         p3         x  x    x   x
    %         p4         x  x    x   x
    
    
    for i = 1:size(pivot_mat(:,2),1)
        l_limit =  max(joint_pos_lower_limit(3)+joint_lim_margin_list(3), coupling_lower_limit(1)-pivot_mat(i,2)+joint_lim_margin_list(3));
        u_limit = min(joint_pos_upper_limit(3)-joint_lim_margin_list(3), coupling_upper_limit(1)-pivot_mat(i,2)-joint_lim_margin_list(3));
        if l_limit >= u_limit
            error(sprintf('Couple Joint2&3 limit  setting error: l_limit: %d should be smaller than u_limit: %d\n', l_limit, u_limit))
        end
        joint3_coup_pivot_mat = [joint3_coup_pivot_mat,...
                          linspace(l_limit, u_limit, joint_pivot_num_list(3))'];

    end
    


    config_mat = joint_all_combs(pivot_mat, joint3_coup_pivot_mat);



    % check if there is any pivot points that is out of joint limits.
    mistakes_count = 0;
    for i = 1:size(config_mat,2)
        if ~hw_joint_space_check(config_mat(:,i).',joint_pos_upper_limit,joint_pos_lower_limit,...
                coupling_index_list,coupling_upper_limit,coupling_lower_limit)
            mistakes_count = mistakes_count+1;
        end
    end
    
    if mistakes_count~=0
        error('mistakes_count %d>0, There is some point out of joint limit. Please rerun the program: run_collectData.m', mistakes_count)
    end


    pivot_mat_safeCheck = pivot_mat([1,end],:); 
    joint3_coup_pivot_mat_safeCheck = joint3_coup_pivot_mat([1,end], [1,end]);
    config_mat_safeCheck = joint_all_combs(pivot_mat_safeCheck, joint3_coup_pivot_mat_safeCheck);
    
    % 
    % pivot_points_path = fullfile(root_path, 'uniform', 'raw_data', ['N', int2str(N)])
    % if ~exist(pivot_points_path, 'dir')
    %    mkdir(pivot_points_path)
    % end
    % save(fullfile(pivot_points_path, 'desired_pivot_points.mat'));

end

function config_mat = joint_all_combs(pivot_mat, joint3_coup_pivot_mat)
    config_mat = [];
    joint_pivot_num = size(pivot_mat,1);
    index_list =      [1,     1,     1,     1,     1,     1]; 
    is_reverse_list = [false, false, false, false, false, false];
    prev_index_list = index_list;
    cnt = 0;
    while true
        cnt = cnt + 1;
        vec =[];
        for i = 1:6
            % toggle reverse status for each joint if index jump from N to 1
            if index_list(i) < prev_index_list(i)
                is_reverse_list(i) = ~is_reverse_list(i);
            end
        end
        
        for i = 1:6          
            % index vector for joint 3
            if i==3
                if is_reverse_list(2)
                    index_2 = joint_pivot_num + 1 - index_list(2);
                else
                    index_2 =index_list(2);
                end
                
                if is_reverse_list(3)
                    index_3 = joint_pivot_num + 1 - index_list(3);
                else
                    index_3 =index_list(3);
                end
                
                vec = [vec;joint3_coup_pivot_mat(index_3, index_2)];

             
            % index vector for other joints
            else
                if is_reverse_list(i)
                    idex_i = joint_pivot_num + 1 - index_list(i);
                else
                    idex_i = index_list(i);
                end
                
                vec = [vec;pivot_mat(idex_i,i)];
            end
        end
        config_mat = [config_mat, vec];
        prev_index_list = index_list;
        
        
        % break loop condition
        if  cnt >= joint_pivot_num^6
            break
        end
        index_list = count_index(index_list, joint_pivot_num);
        disp(cnt)

    end

end

function index_list_output = count_index(index_list_input, overflow_num)
    i = 6;
    index_list_input(i) = index_list_input(i)+1;
    while(index_list_input(i)>overflow_num) 
        index_list_input(i) = 1;
        i = i-1;
        index_list_input(i) = index_list_input(i)+1;
    end
    index_list_output = index_list_input;
end

