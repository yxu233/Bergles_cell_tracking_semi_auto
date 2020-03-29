%% Cell tracking


%% Things to fix:
% (1) disallow ties
% (2) Maybe do watershed as a pre-processing step (so don't waste time
% during analysis)


opengl hardware;
close all;

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
matrix_timeseries = cell(2000, numfids/2);

%% Get first frame
fileNum = 1;
thresh_size = 300; % pixels at the moment
num_slices = 100; % 100 * 3 == 300 microns
[all_s, frame_1, truth_1] = load_data_into_struct(foldername, natfnames, fileNum, all_s, thresh_size, num_slices);

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
for fileNum = 3 : 2: numfids - 2
    % get next frame
    [all_s, frame_2, truth_2] = load_data_into_struct(foldername, natfnames, fileNum, all_s, thresh_size, num_slices);
    
    %% Loop through struct to find nearest neighbors
    % first frame is always taken from "matrix_timeseries" ==> which has
    % been cleaned and sorted
    cur_timeseries = {matrix_timeseries{:, timeframe_idx}};
    array_centroid_indexes = [];
    for idx = 1:length(cur_timeseries)
        if ~isempty(cur_timeseries{idx})
            array_centroid_indexes = [array_centroid_indexes; cur_timeseries{idx}.centroid];
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
    
    
    % find nearest neighbours
    [neighbor_idx, D] = knnsearch(next_centroids, cur_centroids, 'K', 1);
    
    %% Do preliminary loop through to find VERY CONFIDENT neighbours
    %% add into the cell list with corresponding cell number
    % SSIM > 0.4, and distance very small < 10 pixels
    % also assign anythign SUPER LARGE DISTANCE ==> bad
    % leave only the mediocre SSIMs
    
    %% Loop through each neighbor for comparison
    crop_size = 60
    z_size = 16
    ssim_val_thresh = 0.30
    dist_thresh = 15
    upper_dist_thresh = 30
    histogram(D);
    figure(2);
    idx_non_confident = [];
    for check_neighbor = 1:length(neighbor_idx)
        [crop_frame_1, crop_frame_2, crop_truth_1, crop_truth_2, mip_1, mip_2] = crop_centroids(cur_centroids, next_centroids, frame_1, frame_2, truth_1, truth_2, check_neighbor, neighbor_idx, crop_size, z_size);

        %% accuracy metrics
        dist = D(check_neighbor);
        ssim_val = ssim(crop_frame_1, crop_frame_2);
        mae_val = meanAbsoluteError(crop_frame_1, crop_frame_2);
        psnr_val = psnr(crop_frame_1, crop_frame_2);

        title(strcat('ssim: ', num2str(ssim_val), '  dist: ', num2str(dist)))
        
        %% if ssim_val very high AND distance small ==> save the cell
        if ssim_val > ssim_val_thresh && dist < dist_thresh
            %% Plot to verify
            %subplot(1, 2, 1); imshow(mip_1);
            %subplot(1, 2, 2); imshow(mip_2);
            next_cell = next_timeseries(neighbor_idx(check_neighbor));
            voxelIdxList = next_cell.objDAPI;
            centroid = next_cell.centerDAPI;
            cell_num = check_neighbor;
            % create cell object
            cell_obj = cell_class(voxelIdxList,centroid, cell_num);
            matrix_timeseries{check_neighbor, timeframe_idx + 1} = cell_obj;
            %pause
            
            %% also eliminate based on upper boundary
        elseif dist > upper_dist_thresh
            continue
        else
            idx_non_confident = [idx_non_confident, check_neighbor];
        end
    end
    
    
    
    
    %% Loop through NON-CONFIDENT ONES for comparison
    % first find index of all non-confident ones
    
    close all;
    figure(3);
    for idx_nc = 1:length(idx_non_confident)
        check_neighbor = idx_non_confident(idx_nc);
        
        [crop_frame_1, crop_frame_2, crop_truth_1, crop_truth_2, mip_1, mip_2, crop_blank_truth_1, crop_blank_truth_2] = crop_centroids(cur_centroids, next_centroids, frame_1, frame_2, truth_1, truth_2, check_neighbor, neighbor_idx, crop_size, z_size);
        
        %% manual correction
        [option_num, matrix_timeseries] = Bergles_manual_correct(crop_frame_1, crop_frame_2, crop_truth_1, crop_truth_2, crop_blank_truth_1, crop_blank_truth_2...
                                            ,frame_1, frame_2, truth_1, truth_2...
                                            , mip_1, mip_2, D, check_neighbor, neighbor_idx...
                                            , matrix_timeseries, cur_timeseries, next_timeseries, timeframe_idx);



        %% For manual correction ==> also need to know indexes of everything
        %% add option for CLAHE?
        %% add option to replot as bigger plot (see more of the surrounding region)
        %% add option to select correct matching cell body
        %% add option to (1) say true/matched or (2) say not matched
        %cc = bwconncomp(truth);
        %stats = regionprops3(cc,'Volume','Centroid', 'VoxelIdxList'); %%%***good way to get info about region!!!
        
        %input_im = return_crop_around_centroid(input_im, crop, y, x, z, crop_size, z_size, height, width, depth)
      
   
        close all;
       
    end
    
    %% Identify remaining unassociated cells and add them to the cell list with NEW numbering (at the end of the list)
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
              disp('yeet')
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
    
   
    %% set 2nd time frame as 1st time frame for subsequent analysis
    timeframe_idx = timeframe_idx + 1;
    frame_1 = frame_2;
    truth_1 = truth_2;
    
    
    
    
end

%% parse the structs to get same output file as what Cody has!
csv_matrix = [];
for cell_idx = 1:length(matrix_timeseries(:, 1))
    for timeframe = 1:length(matrix_timeseries(1, :))
        if isempty(matrix_timeseries{cell_idx, timeframe})
           continue; 
        end
            
         matrix_timeseries{cell_idx, timeframe};
        cur_cell = matrix_timeseries{cell_idx, timeframe};
        
        volume = length(cur_cell.voxelIdxList);
        centroid = cur_cell.centroid;
        altogether = [cell_idx, timeframe, centroid, volume];
        
        csv_matrix = [csv_matrix; altogether];
        
    end
    
end

writematrix(csv_matrix, 'output.csv') 









