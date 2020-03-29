%% Generates training multipage tiff data from XML file
opengl hardware;
close all;

cur_dir = pwd;
cd(cur_dir);
foldername = uigetdir();   % get directory
addpath(strcat(cur_dir, '\'))  % adds path to functions


cd(foldername);   % switch directories
fnames = dir('*.t*');

namecell=cell(1);
idx = 1;
for i=1:length(fnames)
    namecell{idx,1}=fnames(i).name;
    idx = idx + 1;
end
trialNames = namecell;
numfids = length(trialNames);   %%% divided by 5 b/c 5 files per pack currently
natfnames=natsort(trialNames);


%% Things to parse:
% (1) Use smoothed coordinates
% (2) Find out all points that are connected to each other (i.e. full
% branch)
% (3) Fix dilation sphere at the end
% (4) Generate internode class
% (5) Generate branch point class
% (6) Can delete all things NOT TOUCHING!!! ==> b/c each new seg overlaps
% by minimum 1 pixel!
% (7) DECIDE CROP SIZE??? especially if have higher magnification as well!
% ==> maybe do 100 width ==> so 200 x 200 crops?
% (8) ONLY QUERY FORWARD ==> which means that should NOT ADAPTIVELY CROP
% INSTEAD ==> if hit boundary of image, need to PAD WITH ZEROS, so that
% query remains in center of image!!!

%(9) ***ACTUALLY, forget about front/back seeds, only do FRONTWARDS flowing
%seed!!! ==> might actually help prevent learning to go back down
%things!??? ==> NAHH, don't actually do this yet...

%% Read in images
for fileNum = 1:2:numfids
    cd(foldername);
    filename = natfnames{fileNum};
    %filename = 'd00_MOBPF08_Reg1_singleOL_3_edit3.traces';
    filename_save = strsplit(filename, '_');
    
    %filename_save = filename_save{1};
    tmp =[sprintf('_%s',filename_save{2:end-1})];
    filename_save = strcat(filename_save{1}, tmp);
    
    %cd(cur_dir);
    xmlstruct = parseXML_SingleCell(filename);
    
    
    filename_image = natfnames{fileNum + 1};
    input_im = load_3D_gray(filename_image, natfnames);
    
    % initialize large empty multipage tiffs
    depth = str2num(xmlstruct.imagesize.depth);
    height = str2num(xmlstruct.imagesize.height);
    width = str2num(xmlstruct.imagesize.width);
    
    
    %% Must sort the xmlstruct.paths in a much more different way!
    [xmlstruct.paths(:).startsOn] = deal([]);  % creates new field "startsOn"
    [xmlstruct.paths(:).originalIdx] = deal([]);  % creates new field "startsOn"
    for path_idx = 1:length(xmlstruct.paths)
        if isfield(xmlstruct.paths(path_idx).attribs, 'startson')
            xmlstruct.paths(path_idx).startsOn = str2num(xmlstruct.paths(path_idx).attribs.startson);
            [xmlstruct.paths(path_idx).originalIdx] = str2num(xmlstruct.paths(path_idx).attribs.id);  % subtract 1 because indices here start from 0
        else
            xmlstruct.paths(path_idx).startsOn = -1;
            [xmlstruct.paths(path_idx).originalIdx] = str2num(xmlstruct.paths(path_idx).attribs.id);  % subtract 1 because indices here start from 0
        end
    end
    
    T = struct2table(xmlstruct.paths)
    sorted = sortrows(T, 'startsOn');
    sortedS = table2struct(sorted)
    
    path_seed_indices = find([sortedS.startsOn] == -1);
    
    
    %% Start looping through each seed path to plot it all
    non_seed_idx = max(path_seed_indices) + 1;
    non_seed_idx_val = sortedS(non_seed_idx).originalIdx;
    
    for seed_idx = 1:length(path_seed_indices)
        if seed_idx + 1 < length(path_seed_indices)
            next_idx_val = sortedS(seed_idx + 1).originalIdx;
        else
            next_idx_val = length(xmlstruct.paths);
        end
        
        full_single_path = zeros([height, width, depth]);
        save_linear_indices = cell(0);
        all_indices = [];
        while non_seed_idx_val < next_idx_val
            x = sortedS(non_seed_idx).points.x;
            y = sortedS(non_seed_idx).points.y;
            z = sortedS(non_seed_idx).points.z;
            
            %% EVERYTHING NEEDS TO BE + 1 because imageJ coords start from 0
            x = x + 1;
            y = y + 1;
            z = z + 1;
            
            linear_ind = sub2ind(size(input_im), y, x, z);
            all_indices = [all_indices; linear_ind];
             
            save_linear_indices{end + 1} = linear_ind;
            %linear_ind = [x,y,z];
            
            %% and if index is the FINAL point in name "s01" ect... ==> then set as 3
            full_single_path(linear_ind) = 1;
            
            name = sortedS(non_seed_idx).attribs.name;
            if contains(name, 's')
                paranode_idx = linear_ind(end);
                full_single_path(paranode_idx) = 3;
            end
            non_seed_idx = non_seed_idx + 1;
            non_seed_idx_val = non_seed_idx_val + 1;
        end
        
        %% if index was duplicated, then set to == 2 ==> branch point
        [uniqueA id j] = unique(all_indices,'first');
        indexToDupes = find(not(ismember(1:numel(all_indices),id)));
        duplicates = all_indices(indexToDupes);
        full_single_path(duplicates) = 2;
        

        %% Show maximum projection along axis = 3
        %mip = max(full_single_path, [], 3);
        %figure(); imshow(mip);
        
        %figure(); volshow(im2double(full_single_path));
        
        
        %% Don't need to generate branch-points below, b/c already know where they are
        % val 1 == normal skeleton
        % val 2 == branchpoint (or overlap point)
        % val 3 == paranode   ==> will need to class weight these last 2
        % for training!!!
        
        
        
     
        
        
        %% Now start generating random crops (that contain "combined" image)
        %% random crops must be around areas with sheath/cytosol as input seeds
        % then save the crops as multi-class, where 2nd class is
        % branchpoints
        % make seed for crop every x pixels as you walk down a path
        % make sure to delete anything in image that is coming from a
        % nearby object!!!
        crop_size = 80;
        z_size = 30;
        seed_every = 20;
        for path_idx = 1:length(save_linear_indices)
            path_indices = save_linear_indices{path_idx};
            for small_seed_idx = 1:seed_every:length(path_indices)
                %% if path is very short, then just take midpoint and make crop on either side
                if length(path_indices) < seed_every
                    seed_front_seg = path_indices(1: round(length(path_indices)/2) - 1);
                    seed_mid_point = path_indices(round(length(path_indices)/2));
                    seed_back_seg = path_indices(round(length(path_indices)/2) + 1: end);
                    
                    %% also skip if too short to do another crop
                elseif small_seed_idx + seed_every > length(path_indices)
                    continue;
                    %% Otherwise, if have longer path, must do multiple seed points
                else
                    seed_front_seg = path_indices(1: small_seed_idx + seed_every - 1);
                    seed_mid_point = path_indices(small_seed_idx + seed_every);
                    if small_seed_idx + seed_every + seed_every > length(save_linear_indices)
                        seed_back_seg = path_indices(small_seed_idx + seed_every + 1: end);
                    else
                        seed_back_seg = path_indices(small_seed_idx + seed_every + 1: small_seed_idx + seed_every + seed_every);
                    end
                end
                
                [x, y, z] = ind2sub(size(input_im), seed_mid_point);
                crop_input = crop_around_centroid(input_im, y, x, z, crop_size, z_size, height, width, depth);
                %crop_combined = crop_around_centroid(combined, y, x, z, crop_size, z_size, height, width, depth);
                
                figure(1); volshow(crop_input, 'alphamap', linspace(0,0.3,256)');
                plot_crop_tmp = full_single_path;
                plot_crop_tmp(seed_mid_point) = 5;
                plot_crop_tmp(seed_front_seg) = 4;
                plot_crop_tmp(seed_back_seg) = 6;
                crop_combined = crop_around_centroid(plot_crop_tmp, y, x, z, crop_size, z_size, height, width, depth);
                
                crop_full_path = crop_around_centroid(full_single_path, y, x, z, crop_size, z_size, height, width, depth);

                figure(2); volshow(crop_combined); title('Not cleaned');

                %% Only want to have segmentations up to next branch point ==> limit spatial complexity
                %% at some point should try just keeping whole spatial complexity!
                % (1) Must first find index of branchpoints associated with
                % current segment of interest
                tmp_save_crop_combined = crop_combined;
                front_crop_indices = find(crop_combined == 4);
                back_crop_indices = find(crop_combined == 6);
                
                idx_cur_branch_points = [];
                
                if ~isempty(find(crop_full_path(front_crop_indices) == 2))
                   id = front_crop_indices(find(crop_full_path(front_crop_indices) == 2));
                   idx_cur_branch_points = [idx_cur_branch_points; id];
                end
                
                if ~isempty(find(crop_full_path(back_crop_indices) == 2))
                   id = back_crop_indices(find(crop_full_path(back_crop_indices) == 2));
                   idx_cur_branch_points = [idx_cur_branch_points; id];
                end
                dil_cur_bps = zeros(size(crop_combined));
                dil_cur_bps(idx_cur_branch_points) = 1;
                dil_cur_bps = imdilate(dil_cur_bps, strel('sphere', 6));
                
                % (2) Subtract out ALL branch points in image ==> to make
                % broken disconnected image
                idx_all_bp_cropped = find(crop_full_path == 2);
                
                % dilate this guy out a bit to help with deletion to get
                % rid of sticky skeleton shapes
                dil_bps = zeros(size(crop_combined));
                dil_bps(idx_all_bp_cropped) = 1;
                dil_bps = imdilate(dil_bps, strel('sphere', 5));
                crop_combined(dil_bps == 1) = 0;
                
                % (3) Then add back the branchpoint identified in part #1
                crop_combined(dil_cur_bps == 1) = 2;
                
                % make a binary mask
                binarized_crop = im2double(imbinarize(crop_combined));
                                
                
                % (4) then subtract out the mid-point to make start + end seed segments
                mid_crop_idx = find(crop_combined == 5);
                break_up = binarized_crop;
                break_up(mid_crop_idx) = 0;
                
                % (5) then eliminate any binary objects in the crop that are NOT
                % connected with the start OR end segments!!!
                new_crop = crop_combined;
                cc = bwconncomp(break_up);
                regions = regionprops3(cc, 'VoxelIdxList');
                
                for region_idx = 1:length(regions.VoxelIdxList)
                    voxel_ind = regions.VoxelIdxList{region_idx};
                    max(crop_combined(voxel_ind));
                    if max(crop_combined(voxel_ind)) < 4   % b/c pixel val 3 == paranode
                        new_crop(voxel_ind) = 0;
                    end
                end
                
                
                % (6) then use this "new_crop" that eliminated
                % non-connected objects to mask the original crop
                crop_combined = tmp_save_crop_combined;
                crop_combined(new_crop == 0) = 0;
                new_crop = crop_combined;
                
                %figure(3); volshow(new_crop); title('Cleaned by branchpoint');
                
                
                % (7) Make sure branchpoints and paranodes correlating with
                % these points are re-inserted!

                
                %% To add in rest of branch points, must correlate with dilated ones b/c "new_crop" is
                %% so chopped up that it no longer reaches the original location
                dil_bps = imdilate(dil_bps, strel('sphere', 2));
                add = imadd(imbinarize(new_crop), imbinarize(dil_bps));
                
                cc = bwconncomp(imbinarize(add));
                regions = regionprops3(cc, 'VoxelIdxList');
                
                final_branch_points = zeros(size(new_crop));
                for region_idx = 1:length(regions.VoxelIdxList)
                    voxel_ind = regions.VoxelIdxList{region_idx};
                    max(crop_combined(voxel_ind));
                    if max(crop_combined(voxel_ind)) > 1   % b/c pixel val 3 == paranode
                        final_branch_points(voxel_ind) = 1;
                    end
                end   
                %skel = bwskel(imbinarize(final_branch_points));
                
                idx_new_crop = find(final_branch_points > 0);
                idx_corr_bp = find(crop_full_path(idx_new_crop) == 2);   % index of all branchpoints correlating with new_crop
                idx_corr_bp = idx_new_crop(idx_corr_bp);                
                
                idx_corr_para = find(crop_full_path(idx_new_crop) == 3);  % index of all paranodes correlating with new_crop
                idx_corr_para = idx_new_crop(idx_corr_para);
                
                
                %% Remake "new_crop" by masking out with the branchpoints included
                crop_combined = tmp_save_crop_combined;
                crop_combined(final_branch_points == 0) = 0;
                new_crop = crop_combined;
                
                new_crop(idx_corr_bp) = 2;
                new_crop(idx_corr_para) = 3;
                
                figure(3); volshow(new_crop); title('Cleaned by branchpoint');                                

                %% now save:
                % (1) Input image crop + crop w/ seed 1 + truth multiclass (everything except
                % seed 1)
                % (2) Input image crop + crop w/ seed 2 + truth multiclass (everything except
                % seed 2)
                seed_1 = zeros(size(new_crop));
                seed_1(new_crop == 6) = 1;
                % skip seed_1 if too small:
                skip_seed_1 = 0;
                if length(find(seed_1 == 1)) < 4
                    skip_seed_1 = 1;
                end
                seed_1_dil = imdilate(seed_1, strel('sphere', 2));
                figure(4); volshow(im2double(seed_1)); title('Cleaned by branchpoint');
  
                %% 2nd seed
                seed_2 = zeros(size(new_crop));
                seed_2(new_crop == 4) = 1;                
                % ELIMINATE ANY SEEDS THAT ARE TOO SMALL
                skip_seed_2 = 0;
                if length(find(seed_2 == 1)) < 4
                    skip_seed_2 = 1;
                end
                seed_2_dil = imdilate(seed_2, strel('sphere', 2));
                figure(8); volshow(im2double(seed_2)); title('Cleaned by branchpoint');

                
                %% ONLY QUERY FORWARD:
                % (1) Find mid-point and subtract out of image to divide 2
                % sides
                tmp_new_crop = new_crop;
                mid_point = find(new_crop == 5);
                new_crop(mid_point) = 0;
                
                % (2) Make binary and then region props to find 2 region
                % sides
                % make a binary mask
                binarized_crop = im2double(imbinarize(new_crop));

                cc = bwconncomp(binarized_crop);
                % Delete anything too small
                for cc_idx = 1:length(cc.PixelIdxList)
                   if length(cc.PixelIdxList{cc_idx}) < 2
                       cc.PixelIdxList{cc_idx} = [];
                   end
                end
                regions = regionprops3(cc, 'VoxelIdxList');
                
                truth_1_class_1 = zeros(size(new_crop));
                truth_2_class_1 = zeros(size(new_crop));
                
                % (3) ***make sure to associate correct region to correct
                % truth class!
                for reg_idx = 1:length(regions.VoxelIdxList)
                   cur_region = regions.VoxelIdxList{reg_idx};
                   if find(seed_1(cur_region) == 1)
                       truth_2_class_1(cur_region) = 1;  
                   elseif find(seed_2(cur_region) == 1)
                       truth_1_class_1(cur_region) = 1;
                   end
                end
                
                % Catch exception, where one of the traces is skipped!
                if skip_seed_1 == 1
                    %truth_1_class_1(imbinarize(seed_2)) = 0;
                    truth_1_class_1 = zeros(size(new_crop));  % set other channel to 0
                    %truth_2_class_1(cur_region) = 1;
                    %truth_2_class_1(imbinarize(seed_2)) = 0;
                end
                if skip_seed_2 == 1
                    truth_2_class_1 = zeros(size(new_crop));  % set other channel to 0
                    %truth_1_class_1(cur_region) = 1;
                    %truth_1_class_1(imbinarize(seed_1)) = 0;
                end
                
                
                truth_1_class_1(mid_point) = 1;
                truth_1_class_1(seed_1 == 1) = 0;    % to be safe, subtract out the seed one more time
                figure(5); volshow(im2double(truth_1_class_1)); title('Cleaned by branchpoint');
                truth_1_class_1_dil = imdilate(truth_1_class_1, strel('sphere', 2));
 
                
                truth_2_class_1(mid_point) = 1;
                truth_2_class_1(seed_2 == 1) = 0;    % to be safe, subtract out the seed one more time
                figure(9); volshow(im2double(truth_2_class_1)); title('Cleaned by branchpoint');
                truth_2_class_1_dil = imdilate(truth_2_class_1, strel('sphere', 2));

                %if length(regions.VoxelIdxList) > 2
                %    error('Too many possible queries');
                %end     
                
                %% Also skip if the truth images contain nothing
                if length(find(truth_1_class_1 == 1)) < 4
                    skip_seed_1 = 1;
                end
                if length(find(truth_2_class_1 == 1)) < 4
                    skip_seed_2 = 1;
                end
                

                %% CLASS 2 and CLASS 3 also need to have separate outputs now...
                %% ALSO ==> need to figure out a way to connect ends to branchpoints that are cut out
                
                

                %% class 2 == branchpoints
                truth_1_class_2 = imbinarize(truth_1_class_1);
                truth_1_class_2(new_crop ~= 2) = 0;
                truth_1_class_2_dil = imdilate(truth_1_class_2, strel('sphere', 2));

                figure(6); volshow(im2double(truth_1_class_2_dil)); title('Cleaned by branchpoint');
                
                 %% class 3 == paranodes
                truth_1_class_3 = imbinarize(truth_1_class_1);
                truth_1_class_3(new_crop ~= 3) = 0;
                truth_1_class_3_dil = imdilate(truth_1_class_3, strel('sphere', 4));
                
                figure(7);  volshow(im2double(truth_1_class_3_dil)); title('Cleaned by branchpoint');
                %mip = max(truth_1_class_3_dil, [], 3);
                %subplot(5, 2, 7); imshow(mip);
                
                %% For 2nd truth image
                %% class 2 == branchpoints
                truth_2_class_2 = imbinarize(truth_2_class_1);
                truth_2_class_2(new_crop ~= 2) = 0;
                truth_2_class_2_dil = imdilate(truth_2_class_2, strel('sphere', 2));

                figure(10); volshow(im2double(truth_2_class_2_dil)); title('Cleaned by branchpoint');
                
                 %% class 3 == paranodes
                truth_2_class_3 = imbinarize(truth_2_class_1);
                truth_2_class_3(new_crop ~= 3) = 0;
                truth_2_class_3_dil = imdilate(truth_2_class_3, strel('sphere', 4));
                
                figure(11); volshow(im2double(truth_2_class_3_dil)); title('Cleaned by branchpoint');
 
                
                
                
                
                %% Save training data
                %crop=crop-min(crop(:)); % shift data such that the smallest element of A is 0
                %crop=crop/max(crop(:)); % normalize the shifted data to 1
                cd(foldername);
                cd('./save outputs cropped DILATED');
                
                %figure(1); volshow(crop_input);
                for k = 1:z_size
                    
                    %figure(2); imshow(input);
                    if ~skip_seed_1
                        input = crop_input(:, :, k);
                        input = adapthisteq(input);
                        
                        %% save input images
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_start_', int2str(small_seed_idx), '_input_crop.tif') , 'writemode', 'append', 'Compression','none')
                        
                        %% save CLAHE input images
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_start_', int2str(small_seed_idx), '_CLAHE_input_crop.tif') , 'writemode', 'append', 'Compression','none')
                        
                        % save cropped seeds
                        input = seed_1_dil(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_start_', int2str(small_seed_idx), '_DILATE_seed_crop.tif') , 'writemode', 'append', 'Compression','none')
                        
                        % save truth images
                        input = truth_1_class_1_dil(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_start_', int2str(small_seed_idx), '_DILATE_truth_class_1_crop.tif') , 'writemode', 'append', 'Compression','none')
                        
                        input = truth_1_class_2_dil(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_start_', int2str(small_seed_idx), '_DILATE_truth_class_2_crop.tif') , 'writemode', 'append', 'Compression','none')

                        input = truth_1_class_3_dil(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_start_', int2str(small_seed_idx), '_DILATE_truth_class_3_crop.tif') , 'writemode', 'append', 'Compression','none')

                        %% NON-DILATED:
                        % save cropped seeds
                        input = seed_1(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_start_', int2str(small_seed_idx), '_seed_crop_start.tif') , 'writemode', 'append', 'Compression','none')
                        
                        % save truth images
                        input = truth_1_class_1(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_start_', int2str(small_seed_idx), '_truth_class_1_crop.tif') , 'writemode', 'append', 'Compression','none')
                        
                        input = truth_1_class_2(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_start_', int2str(small_seed_idx), '_truth_class_2_crop.tif') , 'writemode', 'append', 'Compression','none')

                        input = truth_1_class_3(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_start_', int2str(small_seed_idx), '_truth_class_3_crop.tif') , 'writemode', 'append', 'Compression','none')

                                                
                    end

                    
                    
                    if ~skip_seed_2
                        %% save input images
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_end_', int2str(small_seed_idx), '_input_crop.tif') , 'writemode', 'append', 'Compression','none')
                        
                        %% save CLAHE input images
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_end_', int2str(small_seed_idx), '_CLAHE_input_crop.tif') , 'writemode', 'append', 'Compression','none')
                                                
                        % save cropped seeds
                        input = seed_2_dil(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_end_', int2str(small_seed_idx), '_DILATE_seed_crop.tif') , 'writemode', 'append', 'Compression','none')
                        
                        % save truth images
                        input = truth_2_class_1_dil(:, :, k);                       
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_end_', int2str(small_seed_idx), '_DILATE_truth_class_1_crop.tif') , 'writemode', 'append', 'Compression','none')
                        
                        input = truth_2_class_2_dil(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_end_', int2str(small_seed_idx), '_DILATE_truth_class_2_crop.tif') , 'writemode', 'append', 'Compression','none')

                        input = truth_2_class_3_dil(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_end_', int2str(small_seed_idx), '_DILATE_truth_class_3_crop.tif') , 'writemode', 'append', 'Compression','none')

                        
                        %% NON-DILATED:
                        % save cropped seeds
                        input = seed_2(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_end_', int2str(small_seed_idx), '_seed_crop.tif') , 'writemode', 'append', 'Compression','none')
                        
                        % save truth images
                        input = truth_2_class_1(:, :, k);                       
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_end_', int2str(small_seed_idx), '_truth_class_1_crop.tif') , 'writemode', 'append', 'Compression','none')
 
                        input = truth_2_class_2(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_end_', int2str(small_seed_idx), '_truth_class_2_crop.tif') , 'writemode', 'append', 'Compression','none')

                        input = truth_2_class_3(:, :, k);
                        imwrite(input, strcat(filename_save,'_path_', int2str(i), '_', int2str(path_idx), '_seed_end_', int2str(small_seed_idx), '_truth_class_3_crop.tif') , 'writemode', 'append', 'Compression','none')

                           
                        
                    end
                    
                    
                end
                cd(cur_dir)
                skip_seed_1
                skip_seed_2
            end
        end
    end
end



function crop = crop_around_centroid(input_im, y, x, z, crop_size, z_size, height, width, depth)
box_x_max = x + crop_size; box_x_min = x - crop_size;
box_y_max = y + crop_size; box_y_min = y - crop_size;
box_z_max = z + z_size/2; box_z_min = z - z_size/2;

im_size_x = height;
im_size_y = width;
im_size_z = depth;

if box_x_max > im_size_x
    overshoot = box_x_max - im_size_x;
    box_x_max = box_x_max - overshoot;
    box_x_min = box_x_min - overshoot;
end

if box_x_min <= 0
    overshoot_neg = (-1) * box_x_min + 1;
    box_x_min = box_x_min + overshoot_neg;
    box_x_max = box_x_max + overshoot_neg;
end


if box_y_max > im_size_y
    overshoot = box_y_max - im_size_y;
    box_y_max = box_y_max - overshoot;
    box_y_min = box_y_min - overshoot;
end

if box_y_min <= 0
    overshoot_neg = (-1) * box_y_min + 1;
    box_y_min = box_y_min + overshoot_neg;
    box_y_max = box_y_max + overshoot_neg;
end



if box_z_max > im_size_z
    overshoot = box_z_max - im_size_z;
    box_z_max = box_z_max - overshoot;
    box_z_min = box_z_min - overshoot;
end

if box_z_min <= 0
    overshoot_neg = (-1) * box_z_min + 1;
    box_z_min = box_z_min + overshoot_neg;
    box_z_max = box_z_max + overshoot_neg;
end

box_x_max - box_x_min
box_y_max - box_y_min
box_z_max - box_z_min


crop = input_im(box_x_min:box_x_max - 1, box_y_min:box_y_max - 1, box_z_min:box_z_max - 1);
end









