%% Cell tracking
%% Important functional notes about updates:
% (1) There is now a failsafe after manual correction where you correct any
% cells that are still merged to more than 1 cell. Things to note about
% this stage:
%           - the GUI is identical to the other stage
%           - at the end, you will be asked if you want to repeat this
%           checking. You can do so until there are 0 merged cells.
%           - the number of cells to check may be higher than anticipated.
%           This is because you have to check 2 cells per merge (i.e. the
%           cell that merged incorrectly, and the cell that was correctly
%           associated).
% (2) One error that can now be fixed is when nearby cells are segmented
% together and not separated by the CNN. We could try to fix this with
% watershed as well, but I think it's easier to just do it in the GUI. TO
% fix these errors, simply:
%           - place a new dot using "3" on the right frame. This can even
%           be inside the segmentation of the wrongly connected cells,
%           because as long as you place a dot, you create a "new cell"
%           that is now separate from the wrongly connected ones.
%          
% (3) *** remember that you can now check on your work by pressing
% "backspace"!!! super helpful
%
% (4) I also added a hotkey "n" so you can add cells on the left frame now.
% It will delete the position of the current cell on the left frame, and
% you are free to choose a new position. This wasn't that useful in the
% end...
%
% (4) please use "3" to add new cells and not "a"
%
% (5) ***FYI, the reason some cells near the last few slices may not be
% "detected" after the analysis is because I actually delete them. I do
% this because I find that the rigid cut-off near the bottom is never that
% satisfactory, as there are so many cells, and chances are high that one
% frame will get cut-off higher than another, leading to a ton of
% unassociated cells. I essentially just track cells down to a certain # of
% frames. If you want to keep those deeper cells, then choose a lower value
% in the first GUI that prompts for "last_slice". ***feel free to ask me
% more if you're curious.
%
% (6) The distance thresholds are now automatically set for each dataset to
% the 90th and 95th percentile for non-confident cell triaging.
%
% (7) There's also a plot of the vectors of each region in the manual
% correction GUI now. I find that this helps sometimes to know how extreme
% the movement is. I generally don't look at it all that much, but
% sometimes if it's at a 90 deg angle, it might give you a hint to look
% harder at the 2 cells directly ontop of each other.


%% Last things to do:
% speed up the plotting because still a bit slow

%% New additions: version 1.3
% (1) Non-cell centered figures + green/red overlay?  ==> DONE 
% (2) Counter for # of cells left to check ==> DONE 
% (3) Add timepoint # on top of images ==> DONE
% (4) Change color scheme on top image stackes for data/seg ==> to
% white/red or green/red
% (5) REMOVE FINAL NEW CELL CHECKER

%% More updates:
% (1) added vector analysis
% (2) adjusted threshold for SSIM to be more inclusive
% (3) distance thresholds now auto discovered (90th, 95th outliers)
% (4) added XZ projections
% (5) added depth indicator on bottom left (as well as cur cell and frame #)
% (6) removed small crop size... but maybe shouldn't have??? ==> DONE

%% Manual correction keys:
% 1 == yes, is matched
% 2 == no, not matched
% 3 == add new point in any arbitrary location

% a == "add" different associated cell  *** deprecated, use "3" instead
% s == "scale" image to new dimensions (to zoom in/out)
% d == "delete" current cell on current timeframe (b/c it's garbage and not a real cell
% c == "clahe" enhances intensity with CLAHE
% n == "new" cell on the LEFT frame (deletes old cell and prompts for new
% user selected cell point
% BACKSPACE == return to previous frame of correction
% ESC == breaks the loop exits the GUI

%% GUI notes:
% (1) Top slice viewers
% (2) Bottom max projections
% - green is loc of cell on left frame
% - magenta is loc of cell on right frame
% - blue is trail of past locations
% (3) Middle XZ max projections  ***note, is proj of +/- 30 px in X dim
% - top is max proj of left frame
% - bottom is max proj of right frame
%
% (4) Plot of vectors of movement for each cell between frames
% large green line is the average vector, and the thickest vector is the
% current cell of interest

opengl hardware;
close all;

addpath('./IO_func/')
addpath('./man_corr_func/')
addpath('./watershed_func/')
addpath('./cell_crop_func/')

cur_dir = pwd;
addpath(strcat(cur_dir))  % adds path to functions
cd(cur_dir);

%% Initialize
foldername = uigetdir();   % get directory

%% Run Analysis
cd(foldername);   % switch directories
nameCat = '*tif*';
fnames = dir(nameCat);

trialNames = {fnames.name};
numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently

cd(cur_dir);
natfnames=natsort(trialNames);

%% Read in images
empty_file_idx_sub = 0;
all_s = cell(0);
matrix_timeseries = cell(10000, numfids/2);

%% Input dialog values
prompt = {'crop size (XY px): ', 'z_size (Z px): ', 'ssim_thresh (0 - 1): ', 'low_dist_thresh (0 - 20): ', 'upper_dist_thresh (30 - 100): ', 'min_siz (0 - 500): ', 'first_slice: ', 'last_slice: ', 'manual_correct? (Y/N): '};
dlgtitle = 'Input';
definput = {'200', '20', '0.25', '20', '30', '10', '1', '130', 'Y'};
%definput = {'200', '20', '0.30', '15', '25', '50', '5', '120', 'Y'};
%definput = {'200', '20', '0.30', '15', '25', '50', '5', '120', 'Y'};

%% Switched to dist_thresh == 20 from 15 for scaled!!! and upper dist thresh from 20 to 30

answer = inputdlg(prompt,dlgtitle, [1, 35], definput);

crop_size = str2num(answer{1})/2;
z_size = str2num(answer{2});
ssim_val_thresh = str2num(answer{3});
dist_thresh = str2num(answer{4});
upper_dist_thresh = str2num(answer{5});

thresh_size = str2num(answer{6}); % pixels at the moment
first_slice = str2num(answer{7});
last_slice = str2num(answer{8}); % 100 * 3 == 300 microns
manual_correct_bool = answer{9};

%% Get first frame
fileNum = 1;
[all_s, frame_1, truth_1, og_size] = load_data_into_struct(foldername, natfnames, fileNum, all_s, thresh_size, first_slice, last_slice);


%% save all objects in first frame as list of cells
% maybe need to add way to delete out bad objects here as well???
timeframe_idx = 1;
for cell_idx = 1:length(all_s{timeframe_idx})
    cur_s = all_s{timeframe_idx};
    
    voxelIdxList = cur_s(cell_idx).objDAPI;
    centroid = cur_s(cell_idx).centerDAPI;
    cell_num = cell_idx;
    % create cell object
    cell_obj = cell_class(voxelIdxList,centroid, cell_num);
    matrix_timeseries{cell_idx} = cell_obj;
end

%% Set total_cells variable to keep track of head of list
total_cells = length(all_s{1});

%% get subesquent frame
for fileNum = 3 : 2: numfids
    
    %% Also eliminate if on edges of crop??? Either XY +/- 2 or Z +/- 2
    del_num = 0;
    im_size = size(frame_1);
    height = im_size(1);  width = im_size(2); depth = im_size(3);
    for j = 1:length(matrix_timeseries(:, 1))
        if isempty(matrix_timeseries{j, timeframe_idx})
            continue;
        end
        cur_cell = matrix_timeseries{j, timeframe_idx};
        z_p = round(cur_cell.centroid(3));
        x_p = round(cur_cell.centroid(2));
        y_p = round(cur_cell.centroid(1));
        % delete if z_position of centroid within top 5 frames
        if z_p > (last_slice - first_slice) - 2 || x_p <= 2 || x_p > height - 2 || y_p <= 2 || y_p > width -2
            matrix_timeseries(j, :) = {[]};
            del_num = del_num + 1;
            %disp(num2str(del_num));
        end
    end
    
    
    % get next frame
    [all_s, frame_2, truth_2, og_size] = load_data_into_struct(foldername, natfnames, fileNum, all_s, thresh_size, first_slice, last_slice);
    
    %% Loop through struct to find nearest neighbors
    % first frame is always taken from "matrix_timeseries" ==> which has
    % been cleaned and sorted
    cur_timeseries = {matrix_timeseries{:, timeframe_idx}};
    array_centroid_indexes = [];
    for idx = 1:length(cur_timeseries)
        if ~isempty(cur_timeseries{idx})
            array_centroid_indexes = [array_centroid_indexes; cur_timeseries{idx}.centroid];
        else
            array_centroid_indexes = [array_centroid_indexes; [nan, nan, nan]];
        end
    end
    cur_centroids = array_centroid_indexes;
    
    % 2nd frame is taken from the unsorted "all_s"
    next_timeseries = all_s{timeframe_idx + 1};
    array_centroid_indexes = [];
    for idx = 1:length(next_timeseries)
        array_centroid_indexes = [array_centroid_indexes; next_timeseries(idx).centerDAPI];
    end
    next_centroids = array_centroid_indexes;
    
    %% Use scaled matrix for nearest neighbor analysis
    cur_centroids_scaled = cur_centroids;
    cur_centroids_scaled(:, 1) = cur_centroids(:, 1) * 0.83;
    cur_centroids_scaled(:, 2) = cur_centroids(:, 2) * 0.83;
    cur_centroids_scaled(:, 3) = cur_centroids(:, 3) * 3;
    
    next_centroids_scaled = next_centroids;
    next_centroids_scaled(:, 1) = next_centroids(:, 1) * 0.83;
    next_centroids_scaled(:, 2) = next_centroids(:, 2) * 0.83;
    next_centroids_scaled(:, 3) = next_centroids(:, 3) * 3;
    
    % find nearest neighbours
    [neighbor_idx, D] = knnsearch(next_centroids_scaled, cur_centroids_scaled, 'K', 1);
    
    %% ***use this for vec analysis later
    % (1) First create a matrix with all cells given idx corresponding
    % to knn analysis so can find them in a crop
    dist_label_idx_matrix = zeros(size(frame_1));
    for dist_idx = 1:length(cur_timeseries)
        if ~isempty(cur_timeseries{dist_idx})
            dist_label_idx_matrix(cur_timeseries{dist_idx}.voxelIdxList) = dist_idx;
        end
    end
    
    %% Find GLOBAL vectors
    plot_bool = 1; 
    if plot_bool
        figure(); hold on;
    end
    [avg_vec, all_unit_v, all_dist_to_avg, cells_in_crop, all_dist_to_avg_RAW] = find_avg_vectors_GLOBAL(dist_label_idx_matrix, cur_centroids_scaled, next_centroids_scaled,  neighbor_idx, plot_bool);

%     z_distance = abs(all_dist_to_avg_RAW(:, 3));
%     outliers = find(isoutlier(z_distance, 'percentiles', [0, 90]));
%     z_outlier_dist = mean(z_distance(outliers));
    
    
    
    %% Adaptively find distance thresholds using outlier analysis
    % (4) check if it is an outlier to the 90th percentile:
    outliers = find(isoutlier(D, 'percentiles', [0, 85]));
    dist_thresh = mean(D(outliers));
    %dist_thresh = 10000;
    
    outliers = find(isoutlier(D, 'percentiles', [0, 95]));
    upper_dist_thresh = mean(D(outliers));
    
    outliers = find(isoutlier(D, 'percentiles', [0, 10]));
    lower_dist_thresh = mean(D(outliers));
        
    %% (1) Do preliminary loop through to find VERY CONFIDENT neighbours
    %% add into the cell list with corresponding cell number
    % SSIM > 0.4, and distance very small < 10 pixels
    % also assign anythign SUPER LARGE DISTANCE ==> bad
    % leave only the mediocre SSIMs
    
    % Loop through each neighbor for comparison
    disp('finding confidently matched and non-matched cells')
    smaller_crop_size = crop_size;
    smaller_z_size = z_size;
    %figure;
    histogram(D);
    figure(2);
    idx_non_confident = [];
    num_matched = 0;
    num_far = 0;
    for check_neighbor = 1:length(neighbor_idx)
        close all;
        if isnan(D(check_neighbor))   % skip all the "nans"
            continue;
        end
        [crop_frame_1, crop_frame_2, crop_truth_1, crop_truth_2, mip_1, mip_2, crop_blank_truth_1, crop_blank_truth_2] = crop_centroids(cur_centroids, next_centroids, frame_1, frame_2, truth_1, truth_2, check_neighbor, neighbor_idx, smaller_crop_size, smaller_z_size);
        
        %% accuracy metrics
        dist = D(check_neighbor);
        ssim_val = ssim(crop_frame_1, crop_frame_2);
        mae_val = meanAbsoluteError(crop_frame_1, crop_frame_2);
        psnr_val = psnr(crop_frame_1, crop_frame_2);
        
        plot_bool = 0;
        skip = 1;
        % skip this if less than 5 cells to get vectors from
        [avg_vec, all_unit_v, all_dist_to_avg, cells_in_crop] = find_avg_vectors(dist_label_idx_matrix...
            , cur_timeseries ,frame_1, crop_size, z_size, cur_centroids...
            ,cur_centroids_scaled, next_centroids_scaled...
            , check_neighbor, neighbor_idx, plot_bool, skip);
        
        % (3) get current vector
        outlier_vec_bool = [];
        if skip && length(cells_in_crop) > 5
            cell_of_interest =  cur_centroids_scaled(check_neighbor, :);
            neighbor_of_cell =  next_centroids_scaled(neighbor_idx(check_neighbor), :);
            vector = abs(cell_of_interest) - abs(neighbor_of_cell);
            if plot_bool
                plot3([0, vector(1)], [0, vector(2)], [0, vector(3)], 'LineWidth', 10);
            end
            dist_to_avg = abs(avg_vec) - abs(vector);
            dist_to_avg = norm(dist_to_avg);
            
            % (4) check if it is an outlier to the 90th percentile:
            outliers = find(isoutlier(all_dist_to_avg, 'percentiles', [0, 90]));
            outliers_idx = cells_in_crop(outliers);
            outlier_vec_bool = find(ismember(outliers_idx, check_neighbor));
        end
        
        
        %% Check z_distance:
%         frame_1_centroid = cur_centroids(check_neighbor, :);
%         frame_2_centroid = next_centroids(neighbor_idx(check_neighbor), :);
%         
%         % get frame 1 crop
%         y = round(frame_1_centroid(1)); x = round(frame_1_centroid(2)); z = round(frame_1_centroid(3));
%         y_1 = round(frame_2_centroid(1)); x_2 = round(frame_2_centroid(2)); z_2 = round(frame_2_centroid(3));
%         
        
        %z_dist_diff = abs(z - z_2);
        

        %% If super far away, just discard the cell
         if dist > upper_dist_thresh
%             num_far = num_far + 1
%             disp(num_far)
             continue
            
            %% if ssim_val very high AND distance small ==> save the cell
         elseif ssim_val >= ssim_val_thresh && dist <= dist_thresh
            
            if dist >= dist_thresh && z_dist_diff <= z_outlier_dist
                disp('z_threshed okay');
            end
            
            next_cell = next_timeseries(neighbor_idx(check_neighbor));
            voxelIdxList = next_cell.objDAPI;
            centroid = next_cell.centerDAPI;
            cell_num = check_neighbor;
            % create cell object
            cell_obj = cell_class(voxelIdxList,centroid, cell_num);
            matrix_timeseries{check_neighbor, timeframe_idx + 1} = cell_obj;
            %pause
            
            num_matched = num_matched + 1;
            disp(strcat('Number of cells matched:' , num2str(num_matched)))
            
            %% Also more lenient if distance away is super small
        elseif dist <= lower_dist_thresh && ssim_val >= 0.1
            next_cell = next_timeseries(neighbor_idx(check_neighbor));
            voxelIdxList = next_cell.objDAPI;
            centroid = next_cell.centerDAPI;
            cell_num = check_neighbor;
            % create cell object
            cell_obj = cell_class(voxelIdxList,centroid, cell_num);
            matrix_timeseries{check_neighbor, timeframe_idx + 1} = cell_obj;
            
            num_matched = num_matched + 1;
            
            disp(strcat('Number of cells matched:' , num2str(num_matched)))
            %% also eliminate based on upper boundary
            
        else
            idx_non_confident = [idx_non_confident, check_neighbor];
        end
        
        
    end
    
    disp(strcat('upper_dist_thresh: ', {'  '}, num2str(upper_dist_thresh)));
    disp(strcat('dist_thresh: ', {'  '}, num2str(dist_thresh)));
    disp(strcat('lower_dist_thresh: ', {'  '}, num2str(lower_dist_thresh)));
    % disp(strcat('z_outlier_dist: ', {'  '}, num2str(z_outlier_dist)));
    
    %% (2) Loop through NON-CONFIDENT ONES for comparison
    %% cell #36 is blob
    
    % first find index of all non-confident ones
    disp('please correct non-confident cells')
    total_num_frames = numfids/2;
    if manual_correct_bool == 'Y'
        close all;
        figure(3);
        idx_nc = 1;
        while idx_nc <= length(idx_non_confident)
            check_neighbor = idx_non_confident(idx_nc);
            
            %% Get x_min, x_max ect... for crop box limits
            frame_2_centroid = next_centroids(neighbor_idx(check_neighbor), :);
            y_first = round(frame_2_centroid(1)); x_first = round(frame_2_centroid(2)); z_first = round(frame_2_centroid(3));
            im_size = size(frame_2);
            height = im_size(1);  width = im_size(2); depth = im_size(3);
            [crop_frame_2, x_min, x_max, y_min, y_max, z_min, z_max] = crop_around_centroid(frame_2, y_first, x_first, z_first, crop_size, z_size, height, width, depth);
            
            %% manual correction
            cur_cell_idx = idx_nc;
            total_cells_to_correct = length(idx_non_confident);
            [option_num, matrix_timeseries, term] = Bergles_manual_correct(frame_1, frame_2, truth_1, truth_2, crop_frame_2...
                ,D, check_neighbor, neighbor_idx...
                ,matrix_timeseries, cur_timeseries, next_timeseries, timeframe_idx...
                ,x_min, x_max, y_min, y_max, z_min, z_max, crop_size, z_size...
                ,cur_centroids, next_centroids...
                ,dist_thresh, ssim_val_thresh...
                ,cur_cell_idx, total_cells_to_correct, total_num_frames...
                ,cur_centroids_scaled, next_centroids_scaled, dist_label_idx_matrix...
                , x_first, y_first, z_first);
            clf;
            
            %% add back button + full exit
            if term == 10 && idx_nc -1 > 0
                idx_nc = idx_nc - 1;
            elseif term == 10 && idx_nc -1 <= 0
                idx_nc = idx_nc;
            elseif term == 99
                break;  %% FULL EXIT
            else
                idx_nc = idx_nc + 1;
            end
            
        end
    end
    close all;
    
    
    
    
        %% (3) Clean double counted cells
    %% Find all cells that are double matched and correct them
    looks_okay = 'Y';
    while looks_okay == 'Y'
        disp('finding doubles to correct')
        [idx_double_counted] = Bergles_manual_correct_double_count(timeframe_idx + 1, frame_1, truth_1, matrix_timeseries, crop_size, z_size);
        
        %% Loop through struct to find nearest neighbors
        % first frame is always taken from "matrix_timeseries" ==> which has
        % been cleaned and sorted
        cur_timeseries = {matrix_timeseries{:, timeframe_idx}};
        array_centroid_indexes = [];
        for idx = 1:length(cur_timeseries)
            if ~isempty(cur_timeseries{idx})
                array_centroid_indexes = [array_centroid_indexes; cur_timeseries{idx}.centroid];
            else
                array_centroid_indexes = [array_centroid_indexes; [nan, nan, nan]];
            end
        end
        cur_centroids = array_centroid_indexes;
        
        % 2nd frame is taken from the unsorted "all_s"
        next_matrix_time = {matrix_timeseries{:, timeframe_idx + 1}};
        array_centroid_indexes = [];
        
        next_timeseries = all_s{timeframe_idx + 1};
        next_timeseries(:) = [];
        for idx = 1:length(next_matrix_time)
            if ~isempty(next_matrix_time{idx})
                array_centroid_indexes = [array_centroid_indexes; next_matrix_time{idx}.centroid];
                next_timeseries(end + 1).centerDAPI =  next_matrix_time{idx}.centroid;
                next_timeseries(end).objDAPI =  next_matrix_time{idx}.voxelIdxList;
            else
                array_centroid_indexes = [array_centroid_indexes; [nan, nan, nan]];
                next_timeseries(end + 1).Core =  1;
            end
        end
        next_centroids = array_centroid_indexes;
        
        %% Use scaled matrix for nearest neighbor analysis
        cur_centroids_scaled = cur_centroids;
        cur_centroids_scaled(:, 1) = cur_centroids(:, 1) * 0.83;
        cur_centroids_scaled(:, 2) = cur_centroids(:, 2) * 0.83;
        cur_centroids_scaled(:, 3) = cur_centroids(:, 3) * 3;
        
        next_centroids_scaled = next_centroids;
        next_centroids_scaled(:, 1) = next_centroids(:, 1) * 0.83;
        next_centroids_scaled(:, 2) = next_centroids(:, 2) * 0.83;
        next_centroids_scaled(:, 3) = next_centroids(:, 3) * 3;
        
        % find nearest neighbours
        [neighbor_idx, D] = knnsearch(next_centroids_scaled, cur_centroids_scaled, 'K', 1);
        
        %% ***use this for vec analysis later
        % (1) First create a matrix with all cells given idx corresponding
        % to knn analysis so can find them in a crop
        dist_label_idx_matrix = zeros(size(frame_1));
        for dist_idx = 1:length(cur_timeseries)
            if ~isempty(cur_timeseries{dist_idx})
                dist_label_idx_matrix(cur_timeseries{dist_idx}.voxelIdxList) = dist_idx;
            end
        end
        disp('please correct double_counted cells')
        total_num_frames = 1;
        
        %% organize list of cells into correct order
        tmp_idx_double_counted = find(idx_double_counted == timeframe_idx(:, 1) + 1);
        tmp_idx_double_counted = idx_double_counted(tmp_idx_double_counted, :);
        
        uniq_idx = unique(tmp_idx_double_counted);
        if ~isempty(uniq_idx)
            tmp_idx_double_counted = unique(tmp_idx_double_counted(:, 2:3));
        end
        
        if manual_correct_bool == 'Y'
            close all;
            figure(3);
            idx_nc = 1;
            while idx_nc <= length(tmp_idx_double_counted)
                
                check_neighbor = tmp_idx_double_counted(idx_nc);
                %% Get x_min, x_max ect... for crop box limits
                frame_2_centroid = next_centroids(tmp_idx_double_counted(idx_nc), :);
                
                y_first = round(frame_2_centroid(1)); x_first = round(frame_2_centroid(2)); z_first = round(frame_2_centroid(3));
                im_size = size(frame_2);
                height = im_size(1);  width = im_size(2); depth = im_size(3);
                [crop_frame_2, x_min, x_max, y_min, y_max, z_min, z_max] = crop_around_centroid(frame_2, y_first, x_first, z_first, crop_size, z_size, height, width, depth);
                
                
                %% add cell first so can be more adaptive later
                next_cell = next_timeseries(neighbor_idx(check_neighbor));
                voxelIdxList = next_cell.objDAPI;
                centroid = next_cell.centerDAPI;
                cell_num = check_neighbor;
                % create cell object
                cell_obj = cell_class(voxelIdxList,centroid, cell_num);
                matrix_timeseries{check_neighbor, timeframe_idx + 1} = cell_obj;
                
                
                %% manual correction
                cur_cell_idx = idx_nc;
                total_cells_to_correct = length(tmp_idx_double_counted);
                [option_num, matrix_timeseries, term] = Bergles_manual_correct(frame_1, frame_2, truth_1, truth_2, crop_frame_2...
                    ,D, check_neighbor, neighbor_idx...
                    ,matrix_timeseries, cur_timeseries, next_timeseries, timeframe_idx...
                    ,x_min, x_max, y_min, y_max, z_min, z_max, crop_size, z_size...
                    ,cur_centroids, next_centroids...
                    ,dist_thresh, ssim_val_thresh...
                    ,cur_cell_idx, total_cells_to_correct, total_num_frames...
                    ,cur_centroids_scaled, next_centroids_scaled, dist_label_idx_matrix...
                    , x_first, y_first, z_first);
                
                clf;
                
                
                %% add back button + full exit
                if term == 10 && idx_nc -1 > 0
                    idx_nc = idx_nc - 1;
                elseif term == 10 && idx_nc -1 <= 0
                    idx_nc = idx_nc;
                elseif term == 99
                    break;  %% FULL EXIT
                else
                    idx_nc = idx_nc + 1;
                end
                
            end
        end
        close all;
        
        %% Satisfied?
        prompt = {'Repeat check for double cells? (Y/N):'};
        dlgtitle = 'Repeat?';
        definput = {'N'};
        answer = inputdlg(prompt,dlgtitle, [1, 35], definput);
        looks_okay = answer{1};
        
    end


    %% (4) Identify remaining unassociated cells and add them to the cell list with NEW numbering (at the end of the list)
    disp('adding non-matched new cells')
    next_timeseries = all_s{timeframe_idx + 1};
    for cell_idx = 1:length(next_timeseries)
        original_cell = next_timeseries(cell_idx).objDAPI;
        if isempty(original_cell)
            continue;
        end
        matched = 0;
        for sorted_idx = 1:length(matrix_timeseries)
            if isempty(matrix_timeseries{sorted_idx, timeframe_idx + 1})
                continue;
            end
            sorted_cell = matrix_timeseries{sorted_idx, timeframe_idx + 1};
            sorted_cell = sorted_cell.voxelIdxList;
            same = ismember(original_cell, sorted_cell);
            
            %% if matched
            if ~isempty(find(same, 1))
                
                matched = 1;
                break;
            end
        end
        
        %% save cell if NOT matched after sorting, then add as new cell to matrix_timeseries
        if matched == 0
            total_cells = total_cells + 1;
            
            next_cell = next_timeseries(cell_idx);
            voxelIdxList = next_cell.objDAPI;
            centroid = next_cell.centerDAPI;
            cell_num = check_neighbor;
            % create cell object
            cell_obj = cell_class(voxelIdxList,centroid, cell_num);
            matrix_timeseries{total_cells, timeframe_idx + 1} =  cell_obj;
        end
    end
   
    
    %% Nothing below the last 10 frames can be a new cell after the first frame has been tested
    del_num = 0;
    for i = 1:length(matrix_timeseries(1, :))
        for j = 1:length(matrix_timeseries(:, 1))
            if isempty(matrix_timeseries{j, i})
                continue;
            end
            cur_cell = matrix_timeseries{j, i};
            z_position = round(cur_cell.centroid(3));
            
            
            if z_position > (last_slice - first_slice) - 10 && i > 1 && isempty(matrix_timeseries{j, i - 1})
                matrix_timeseries(j, :) = {[]};
                del_num = del_num + 1;
                disp(num2str(del_num));
            end
        end
    end
    
    %% (optional) Double check all the cells in the current timeframe that were NOT associated with stuff
    %% just to verify they are ACTUALLY cells???
    
    %% set 2nd time frame as 1st time frame for subsequent analysis
    timeframe_idx = timeframe_idx + 1;
    frame_1 = frame_2;
    truth_1 = truth_2;
    
    
end



%% also save matrix_timeseries
matrix_timeseries_raw = matrix_timeseries;
save('matrix_timeseries_raw', 'matrix_timeseries_raw');


%% parse the structs to get same output file as what Cody has (raw output)
% subtract 1 from timeframe idx AND from cell_idx to match Cody's output!
csv_matrix = [];
for cell_idx = 1:length(matrix_timeseries(:, 1))
    for timeframe = 1:length(matrix_timeseries(1, :))
        if isempty(matrix_timeseries{cell_idx, timeframe})
            continue;
        end
        
        cur_cell = matrix_timeseries{cell_idx, timeframe};
        
        volume = length(cur_cell.voxelIdxList);
        centroid = cur_cell.centroid;
        
        %% Subtract 1 from timeframe index and cell index to match Cody's output!
        altogether = [cell_idx - 1, timeframe - 1, centroid, volume];
        
        csv_matrix = [csv_matrix; altogether];
        
    end
end
writematrix(csv_matrix, 'output_raw.csv')


%% Additional post-processing of edges and errors
% (A) Eliminate everything that only exists on a single frame
num_frames_exclude = 1;
[matrix_timeseries_cleaned] = elim_untracked(matrix_timeseries, num_frames_exclude, foldername, natfnames, crop_size, z_size, thresh_size, first_slice, last_slice);
matrix_timeseries = matrix_timeseries_cleaned;

% (B) eliminate if located above or below +/- 2 AT THE FIRST CELL POINT
% Also eliminate if on edge of crop
all_volumes = [];
del_num = 0;
for i = 1:length(matrix_timeseries(1, :))
    
    for j = 1:length(matrix_timeseries(:, 1))
        if isempty(matrix_timeseries{j, i})
            continue;
        end
        cur_cell = matrix_timeseries{j, i};
        z_position = round(cur_cell.centroid(3));
        % delete if z_position of centroid within top 5 frames
        if z_position > (last_slice - first_slice) - 2 && (i == 1 || isempty(matrix_timeseries{j, i - 1}))
            matrix_timeseries(j, :) = {[]};
            del_num = del_num + 1;
            disp(num2str(del_num));
        end
    end
end

%% (C) check all NEW cells to ensure they are actually new (excluding the first frame)
%frame_num = 2;
% frame_num  = timeframe_idx - 1;
% for fileNum =numfids-2 : 2: numfids
%     [all_s, frame_1, truth_1, og_size] = load_data_into_struct(foldername, natfnames, fileNum, all_s, thresh_size, first_slice, last_slice);
%     [matrix_timeseries] = Bergles_manual_correct_last_frame(frame_num, frame_1, truth_1, matrix_timeseries, crop_size, z_size);
%     frame_num = frame_num + 1;
% end



%% Get all volumes:
% all_volumes = [];
% for i = 1:length(matrix_timeseries(1, :))
%     for j = 1:length(matrix_timeseries(:, 1))
%         if isempty(matrix_timeseries{j, i})
%             continue;
%         end
%         cur_cell = matrix_timeseries{j, i};
%         all_volumes = [all_volumes; length(cur_cell.voxelIdxList)];
%     end
% end

%% parse the structs to get same output file as what Cody has!
% subtract 1 from timeframe idx AND from cell_idx to match Cody's output!
csv_matrix = [];
for cell_idx = 1:length(matrix_timeseries(:, 1))
    for timeframe = 1:length(matrix_timeseries(1, :))
        if isempty(matrix_timeseries{cell_idx, timeframe})
            continue;
        end
        
        cur_cell = matrix_timeseries{cell_idx, timeframe};
        
        volume = length(cur_cell.voxelIdxList);
        centroid = cur_cell.centroid;
        
        %% Subtract 1 from timeframe index and cell index to match Cody's output!
        altogether = [cell_idx - 1, timeframe - 1, centroid, volume];
        
        csv_matrix = [csv_matrix; altogether];
        
    end
end
writematrix(csv_matrix, 'output.csv')

%% also save matrix_timeseries
save('matrix_timeseries', 'matrix_timeseries');

%% Recreate the output DAPI for each frame with cell number for each (create rainbow image)
list_random_colors = randi([1, 20], [1, length(matrix_timeseries)]);
for timeframe_idx = 1:length(matrix_timeseries(1, :))
    im_frame = zeros(og_size);
    for cell_idx = 1:length(matrix_timeseries(:, 1))
        
        if isempty(matrix_timeseries{cell_idx, timeframe_idx})
            continue;
        end
        
        %         if timeframe_idx + 1 < length(matrix_timeseries) && ~isempty(matrix_timeseries{cell_idx, timeframe_idx + 1})
        %            continue;
        %         end
        
        %% Skip everything that IS persisting (so leaving all the NEW cells)
        %         if timeframe_idx > 1 && ~isempty(matrix_timeseries{cell_idx, timeframe_idx - 1})
        %            continue;
        %         end
        
        cur_cell = matrix_timeseries{cell_idx, timeframe_idx};
        
        voxels = cur_cell.voxelIdxList;
        
        im_frame(voxels) = list_random_colors(cell_idx);
    end
    
    % save one frame
    filename_raw = natfnames{timeframe_idx * 2 - 1};
    z_size = length(im_frame(1, 1, :));
    
    im_frame = uint8(im_frame);
    for k = 1:z_size
        input = im_frame(:, :, k);
        imwrite(input, strcat(filename_raw,'_CORR_SEMI_AUTO.tif') , 'writemode', 'append', 'Compression','none')
    end
end



%% Plot number of new cells and number of old cells at each timepoint
new_cells_per_frame = zeros(1, length(matrix_timeseries(1, :)));
terminated_cells_per_frame = zeros(1, length(matrix_timeseries(1, :)));
num_total_cells_per_frame = zeros(1, length(matrix_timeseries(1, :)));
for timeframe_idx = 1:length(matrix_timeseries(1, :))
    for cell_idx = 1:length(matrix_timeseries(:, 1))
        
        if isempty(matrix_timeseries{cell_idx, timeframe_idx})
            continue;
        end
        cur_cell = matrix_timeseries{cell_idx, timeframe_idx};
        
        % new cell if previous frame empty
        if timeframe_idx > 1 && isempty(matrix_timeseries{cell_idx, timeframe_idx - 1})
            new_cells_per_frame(timeframe_idx) =  new_cells_per_frame(timeframe_idx) + 1;
        end
        
        % terminated cells if next frame empty
        if timeframe_idx + 1 < length(matrix_timeseries(1, :)) && isempty(matrix_timeseries{cell_idx, timeframe_idx + 1})
            terminated_cells_per_frame(timeframe_idx + 1) =  terminated_cells_per_frame(timeframe_idx + 1) + 1;
        end
        
        % number of totl cells per frame
        num_total_cells_per_frame(timeframe_idx) = num_total_cells_per_frame(timeframe_idx) + 1;
        
    end
    
end

figure; bar(new_cells_per_frame); title('New cells per frame');
xlabel('frame number'); ylabel('number of new cells');

figure; bar(terminated_cells_per_frame); title('terminated cells per frame');
xlabel('frame number'); ylabel('number of terminated cells');

figure; bar(num_total_cells_per_frame); title('num TOTAL cells per frame');
xlabel('frame number'); ylabel('num TOTAL cells');









