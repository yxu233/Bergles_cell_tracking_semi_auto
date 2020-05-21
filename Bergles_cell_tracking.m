%% Cell tracking
%% Final notes:
% 1) Best with good window + good registration (almost no correction
% needed) ==> ~90 - 95% sensitivity and precision for matching Cody's segmentations
% 2) with worse windows + blur ==> 85 - 90% sensitivity and precision
% 3) Time to do analysis of 1024 x 1024 x 130 volume is about 1 hour, but
% time increases quite a lot with number of lower layers you add to the
% segmentation. Also, of course, depends on quality of images +
% registration.
% 4) Manual correction is mostly just spent correcting cells in the lower
% 20 slices of the z-stack. This is the most time consuming part. In cases
% where a lot of the lower cells die, the manual correction can be blazing
% fast (like correct only 1 or 2 cells)
%
%
% OVERALL: wayyy better than ILASTIK, and is essentially "fully-auto" for
% the top 100 z-slices on good quality images. Almost 90% of manual
% correction time is spent on correcting cells in lower layers.


%% More updates:
% (1) removed small crop size... but maybe shouldn't have???
% (2) added 2nd z project for left frame
% (3) ***add drawLine function
% (4) ***find vector of movement (average in a region)? as a metric
% (5) *** add depth of current slice
% (6) add vector crop plot to help when manual correcting as well???

% (7) move distance metric up as first thing to check 
% (8) make other metrics more lenient!!!


%%
% (2) Some cells touching need to be separated
% (3) Are you using the "3" or "a" button to add?
% showing trajectory?
% (5) find average trajectory around cell

%% (is cellular debris causing an issue???)


%% New additions: version 1.3
% (1) Non-cell centered figures + green/red overlay?  ==> DONE ==> added extra dot
% color for guidance
% (2) Counter for # of cells left to check ==> DONE ***Note: isn't
% accurate, b/c of 2nd round
% (3) Add timepoint # on top of images ==> DONE
% (4) Change color scheme on top image stackes for data/seg ==> to
% white/red or green/red
% (5) Color code z-projection view to see what has already been tracked
% (6) REMOVE FINAL NEW CELL CHECKERTPP


%% New additions: version 1.2
%(1) added ability to "add" cells - hotkey == 3
%(2) now deletes anything solely on single frame
%(3) also deletes anything in lower part of volume if it does not START
%there (centroid)
%(4) also added manual checkup for points at the end which are started as
%"new" cells

% ***how should we deal with super dim cells with no processes ever?
% ***note: 10x vs. 20x!!!
% ***things are fairly consistent now
% * slightly worse on bad quality window ones (ILASTIK BAD) ==> picking out
% some really really dim stuff...
% * should this output be ablet to load back into the syglass?

%% Manual correction keys:
% 1 == yes, is matched
% 2 == no, not matched11
% 3 == add new point in any arbitrary location

% a == "add" different associated cell
% s == "scale" image to new dimensions (to zoom in/out)
% d == "delete" current cell on current timeframe (b/c it's garbage and not a real cell
% c == "clahe" enhances intensity with CLAHE

%% Notes:
% low SSIM ==> mostly due to shifts in axial location/misalignmnet
% careful that pressing keys sometimes writes onto actual program (so
% delete those markings)

% FIXED BERGLES CELL CROP!!!

%% Things to fix:
% (1) disallow ties? ==> would be much much faster...
% (3) the sliding viewer has clipped off top when scaled
% (4) Keep scaling at x 2 when it is on the hard one
% (5) add way to double check cell bodies (that w
% (6) add output folder
% (7) eliminate cells on border?
% (8) correct cell number for watershed display???
% (9) Include directional scaling (to microns) for distance metrics

% (10) double check everything in last frame that was not associated with
% something in the previous frame

% (11) eliminate everything that only exists on a single (or 2?) frames


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
definput = {'200', '20', '0.30', '20', '30', '10', '1', '130', 'Y'};
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
    
    %% Delete things near edges of image???
    
    
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
    
    
    %% Do preliminary loop through to find VERY CONFIDENT neighbours
    %% add into the cell list with corresponding cell number
    % SSIM > 0.4, and distance very small < 10 pixels
    % also assign anythign SUPER LARGE DISTANCE ==> bad
    % leave only the mediocre SSIMs
    
    %% Loop through each neighbor for comparison
    disp('finding confidently matched and non-matched cells')
    smaller_crop_size = crop_size;
    smaller_z_size = z_size;
    %figure;
    histogram(D);
    figure(2);
    idx_non_confident = [];
    num_matched = 0;
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
        
        plot_bool = 1;
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
            vector = cell_of_interest - neighbor_of_cell;
            unit_v_check = vector/norm(vector);
            plot3([0, unit_v_check(1)], [0, unit_v_check(2)], [0, unit_v_check(3)], 'LineWidth', 10);
            
            dist_to_avg = abs(avg_vec) - abs(unit_v_check);
            dist_to_avg = norm(dist_to_avg);
            
            % (4) check if it is an outlier to the 90th percentile:
            outliers = find(isoutlier(all_dist_to_avg, 'percentiles', [0, 90]));
            outliers_idx = cells_in_crop(outliers);
            
            outlier_vec_bool = find(ismember(outliers_idx, check_neighbor));
            
            outlier_vec_bool

            % cell # 95 was missed
            
        end
        
        %% if ssim_val very high AND distance small ==> save the cell
        if dist > upper_dist_thresh
            continue
            
        elseif ~isempty(outlier_vec_bool)
            idx_non_confident = [idx_non_confident, check_neighbor];
            %% Plot for  debug
            plot_full_figure_debug(frame_1, frame_2, truth_1, truth_2, crop_frame_1, crop_frame_2...
                ,crop_truth_1, crop_truth_2,D, check_neighbor, neighbor_idx...
                ,matrix_timeseries, cur_timeseries, next_timeseries, timeframe_idx...
                ,smaller_crop_size, smaller_z_size...
                ,cur_centroids, crop_blank_truth_1, crop_blank_truth_2, next_centroids);
            pause()
        elseif ssim_val > ssim_val_thresh && dist < dist_thresh
            
            %% Plot for debug
            %             plot_full_figure_debug(frame_1, frame_2, truth_1, truth_2, crop_frame_1, crop_frame_2...
            %                 ,crop_truth_1, crop_truth_2,D, check_neighbor, neighbor_idx...
            %                 ,matrix_timeseries, cur_timeseries, next_timeseries, timeframe_idx...
            %                 ,smaller_crop_size, smaller_z_size...
            %                 ,cur_centroids, crop_blank_truth_1, crop_blank_truth_2, next_centroids);
            %pause()
            
            %% Plot to verify
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
        elseif dist <= 7 && ssim_val >= 0.2
            %             % Plot for debug
            %             plot_full_figure_debug(frame_1, frame_2, truth_1, truth_2, crop_frame_1, crop_frame_2...
            %                 ,crop_truth_1, crop_truth_2,D, check_neighbor, neighbor_idx...
            %                 ,matrix_timeseries, cur_timeseries, next_timeseries, timeframe_idx...
            %                 ,smaller_crop_size, smaller_z_size...
            %                 ,cur_centroids, crop_blank_truth_1, crop_blank_truth_2, next_centroids);
            %             pause()


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
            %             subplot(1, 2, 1); imshow(mip_1);
            %             subplot(1, 2, 2); imshow(mip_2);
            %             title(strcat('NON CONF ssim: ', num2str(ssim_val), '  dist: ', num2str(dist)))
            idx_non_confident = [idx_non_confident, check_neighbor];
        end
        %pause
    end
    
    
    %% Loop through NON-CONFIDENT ONES for comparison
    % first find index of all non-confident ones
    disp('please correct non-confident cells')
    total_num_frames = numfids/2;
    if manual_correct_bool == 'Y'
        close all;
        figure(3);
        for idx_nc = 1:length(idx_non_confident)
            check_neighbor = idx_non_confident(idx_nc);
            
            %% Get x_min, x_max ect... for crop box limits
            frame_2_centroid = next_centroids(neighbor_idx(check_neighbor), :);
            y = round(frame_2_centroid(1)); x = round(frame_2_centroid(2)); z = round(frame_2_centroid(3));
            im_size = size(frame_2);
            height = im_size(1);  width = im_size(2); depth = im_size(3);
            [crop_frame_2, x_min, x_max, y_min, y_max, z_min, z_max] = crop_around_centroid(frame_2, y, x, z, crop_size, z_size, height, width, depth);
            
            
            %% manual correction
            cur_cell_idx = idx_nc;
            total_cells_to_correct = length(idx_non_confident);
            [option_num, matrix_timeseries] = Bergles_manual_correct(frame_1, frame_2, truth_1, truth_2, crop_frame_2...
                ,D, check_neighbor, neighbor_idx...
                ,matrix_timeseries, cur_timeseries, next_timeseries, timeframe_idx...
                ,x_min, x_max, y_min, y_max, z_min, z_max, crop_size, z_size...
                ,cur_centroids, next_centroids...
                ,dist_thresh, ssim_val_thresh...
                ,cur_cell_idx, total_cells_to_correct, total_num_frames);
            
            close all;
        end
    end
    
    %% Identify remaining unassociated cells and add them to the cell list with NEW numbering (at the end of the list)
    disp('adding non-matched new cells')
    for cell_idx = 1:length(next_timeseries)
        original_cell = next_timeseries(cell_idx).objDAPI;
        
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

% (C) check all NEW cells to ensure they are actually new (excluding the
% first frame)
frame_num = 2;
for fileNum = 3 : 2: numfids
    [all_s, frame_1, truth_1, og_size] = load_data_into_struct(foldername, natfnames, fileNum, all_s, thresh_size, first_slice, last_slice);
    [matrix_timeseries] = Bergles_manual_correct_last_frame(frame_num, frame_1, truth_1, matrix_timeseries, crop_size, z_size);
    frame_num = frame_num + 1;
end



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









