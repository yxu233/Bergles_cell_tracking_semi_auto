%% Cell tracking

%% Manual correction keys:
% 1 == yes, is matched
% 2 == no, not matched

% a == "add" different associated cell
% s == "scale" image to new dimensions (to zoom in/out)
% d == "delete" current cell on current timeframe (b/c it's garbage and not a real cell
% c == "clahe" enhances intensity with CLAHE 

%% Notes:
% low SSIM ==> mostly due to shifts in axial location/misalignmnet
% careful that pressing keys sometimes writes onto actual program (so
% delete those markings)1

%% Things to fix:
% (1) disallow ties? ==> would be much much faster...
% (2) add plot WITHOUT the green stuff that obscures cell body
% (3) the sliding viewer has clipped off top when scaled
% (4) Keep scaling at x 2 when it is on the hard one
% (5) add way to double check cell bodies (that w

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
matrix_timeseries = cell(1500, numfids/2);


%% Input dialog values
prompt = {'crop size (XY px): ', 'z_size (Z px): ', 'ssim_thresh (0 - 1): ', 'low_dist_thresh (0 - 20): ', 'upper_dist_thresh (30 - 100): ', 'min_siz (0 - 500): ', 'first_slice: ', 'last_slice: ', 'manual_correct? (Y/N): '};
dlgtitle = 'Input';
definput = {'200', '25', '0.30', '15', '25', '200', '10', '110', 'Y'};
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
[all_s, frame_1, truth_1] = load_data_into_struct(foldername, natfnames, fileNum, all_s, thresh_size, first_slice, last_slice);



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
    % get next frame
    [all_s, frame_2, truth_2] = load_data_into_struct(foldername, natfnames, fileNum, all_s, thresh_size, first_slice, last_slice);
    
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
    
    
    % find nearest neighbours
    [neighbor_idx, D] = knnsearch(next_centroids, cur_centroids, 'K', 1);
    
    %% Do preliminary loop through to find VERY CONFIDENT neighbours
    %% add into the cell list with corresponding cell number
    % SSIM > 0.4, and distance very small < 10 pixels
    % also assign anythign SUPER LARGE DISTANCE ==> bad
    % leave only the mediocre SSIMs
    
    %% Loop through each neighbor for comparison
    smaller_crop_size = 60;
    smaller_z_size = 16;
    histogram(D);
    figure(2);
    idx_non_confident = [];
    for check_neighbor = 1:length(neighbor_idx)
        if isnan(D(check_neighbor))   % skip all the "nans"
           continue;
        end
        [crop_frame_1, crop_frame_2, crop_truth_1, crop_truth_2, mip_1, mip_2] = crop_centroids(cur_centroids, next_centroids, frame_1, frame_2, truth_1, truth_2, check_neighbor, neighbor_idx, smaller_crop_size, smaller_z_size);

        %% accuracy metrics
        dist = D(check_neighbor);
        ssim_val = ssim(crop_frame_1, crop_frame_2);
        mae_val = meanAbsoluteError(crop_frame_1, crop_frame_2);
        psnr_val = psnr(crop_frame_1, crop_frame_2);

        
        
        %% if ssim_val very high AND distance small ==> save the cell
        if ssim_val > ssim_val_thresh && dist < dist_thresh
            %% Plot to verify
            %subplot(1, 2, 1); imshow(mip_1);
            %subplot(1, 2, 2); imshow(mip_2);
            %title(strcat('ssim: ', num2str(ssim_val), '  dist: ', num2str(dist)))
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
            [option_num, matrix_timeseries] = Bergles_manual_correct(frame_1, frame_2, truth_1, truth_2, crop_frame_2...
                ,D, check_neighbor, neighbor_idx...
                , matrix_timeseries, cur_timeseries, next_timeseries, timeframe_idx...
                ,x_min, x_max, y_min, y_max, z_min, z_max, crop_size, z_size...
                ,cur_centroids, next_centroids);
            close all;
        end
    end
    
    %% Identify rem1aining unassociated cells and add them to the cell list with NEW numbering (at the end of the list)
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
   
    %% (optional) Double check all the cells in the current timeframe that were NOT associated with stuff
    %% just to verify they are ACTUALLY cells???
    
    
    %% add in ability to plot WITHOUT the red/green overlay
    
    

    %% set 2nd time frame as 1st time frame for subsequent analysis
    timeframe_idx = timeframe_idx + 1;
    frame_1 = frame_2;
    truth_1 = truth_2;
    

end

%% parse the structs to get same output file as what Cody has!
%% subtract 1 from timeframe idx AND from cell_idx to match Cody's output!

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
    im_frame = zeros(size(frame_1));
    for cell_idx = 1:length(matrix_timeseries(:, 1))

        if isempty(matrix_timeseries{cell_idx, timeframe_idx})
            continue;
        end
        cur_cell = matrix_timeseries{cell_idx, timeframe_idx};
        
        voxels = cur_cell.voxelIdxList;
        cell_number = cur_cell.cell_number;
        
        im_frame(voxels) = list_random_colors(cell_number);
    end
    
    % save one frame
    filename_raw = natfnames{timeframe_idx * 2 - 1};
    z_size = length(im_frame(1, 1, :));
    
    im_frame = uint8(im_frame);
    for k = 1:z_size
        input = im_frame(:, :, k);
        imwrite(input, strcat(filename_raw,'_output_SEMI_AUTO.tif') , 'writemode', 'append', 'Compression','none')
    end
end











