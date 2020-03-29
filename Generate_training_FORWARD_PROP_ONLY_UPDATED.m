%% Things to fix:

%(1) Side capture of sheaths is BAD... figure out way to maybe get stuff
%that colocalizes with ends??? ==> DONE

%(2) only generate non-dilated skeletons and let python script dilate at
%run time

%(3) get rid of realllyyyy tiny segments... unreasonably difficult ==> DONE

%(4) active contours or way to snap mask better to intensity?

%(5) figure out how to resize in z for interpolate so that dilations are
%accurate???

%(6) should you include past branches also in the seed? so not just linear
%seeds??? ==> DONE


%(7) during testing ==> only leave remnants of branch or all???

%(8) add paranodes to help with stopping power???... ==> DONE

%(9) make size 96x96x18? or 80x80x16?

%(10) seed is offset from truth after resize??? or somewhere else?... ==>
%DONE ==> just resize without dilation. Error was with dilation and then
%skeletionization

% (11) Seed should be from FRONT SEG up until MIDPOINT, and then the BACK
% SEG should be combined with the rest of the truth of the segmentation!!!
% ==> DONE


% (12) when it splits and hits a myelin sheath ==> get big branching... do you want
% to keep the opposite branch? Maybe not...? Becuase algorithm will only
% seed single branch area in the future???...


% (13) If truly want to do Paranode segmentation ==> then also allow
% paranodes to take up to the last 4 pixels of any segmentation rather than
% the very last one only?

% (14) Are you sure that depth_limit == 3 is sufficient for always full
% coverage???


% (15) fix naming of "_BRANCHED" and "_pos" labels

%(16) seeds are not cleaned (there are remnants coming out of sides of
%image)

% Generates training multipage tiff data from XML file
opengl hardware;
close all;

cur_dir = pwd;
cd(cur_dir);
foldername = uigetdir();   % get directory
addpath(strcat(cur_dir, '\'))  % adds path to functions
addpath(strcat('./Ridge_filter', '\'))


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
            xmlstruct.paths(path_idx).startsOn = str2num(xmlstruct.paths(path_idx).attribs.startson) + 1;
            [xmlstruct.paths(path_idx).originalIdx] = str2num(xmlstruct.paths(path_idx).attribs.id) + 1;  % add 1 because indices here start from 0
        else
            xmlstruct.paths(path_idx).startsOn = 0;
            [xmlstruct.paths(path_idx).originalIdx] = str2num(xmlstruct.paths(path_idx).attribs.id) + 1;  % add 1 because indices here start from 0
        end
    end
    
    T = struct2table(xmlstruct.paths);
    sorted = sortrows(T, 'startsOn');
    sortedS = table2struct(sorted);
    
    
    
    %% Get scaling in um/pixel:
    original_scaling = str2num(xmlstruct.samplespacing.x);
    target_scale = 0.20;
    
    scale_factor = original_scaling/target_scale;
    
    
    %% Loop through everything in sortedS only ONCE
    originalIdx = 1;   % to start things off, first index is always zero
    qc_counter = 0;
    for loop_idx = 1:length([sortedS.originalIdx])
        
        next_loop_idx = find([sortedS.originalIdx] == originalIdx);
        
        if isempty(next_loop_idx)    % SKIP IF FOR SOME REASON an index doesn't exist???
            originalIdx = originalIdx + 1;
            continue;
        end
        
        %% Always plot current index + all branchs that "start on" current index
        full_single_path = zeros([height, width, depth]);
        all_indices = [];
        save_linear_indices = cell(0);
        
        %% (1) Plots current index segment
        x = sortedS(next_loop_idx).points.x;
        y = sortedS(next_loop_idx).points.y;
        z = sortedS(next_loop_idx).points.z;
        % EVERYTHING NEEDS TO BE + 1 because imageJ coords start from 0
        x = x + 1;
        y = y + 1;
        z = z + 1;
        linear_ind = sub2ind(size(input_im), y, x, z);
        all_indices = [all_indices; linear_ind];
        save_linear_indices{end + 1} = linear_ind;
        full_single_path(linear_ind) = 1;
        
        % Also check for paranodes
        name = sortedS(next_loop_idx).attribs.name;
        if contains(name, 's')
            paranode_idx = linear_ind(end);
            full_single_path(paranode_idx) = 3;
        end
        
        %% (2) finds and then plots all branches (if they exist) and recurse backwards
        depth_limit = 3;
        
        [full_single_path] = recurse_branch_tree_backwards(originalIdx, full_single_path, sortedS, depth_limit, size(input_im), save_linear_indices);

        [full_single_path] = recurse_branch_tree(originalIdx, full_single_path, sortedS, depth_limit, size(input_im), save_linear_indices);
        
        
        %% Plot images of current branch and recursive tree to quality check the images
        if loop_idx < 20
            cd(foldername)
            cd('./mip quality check');
            full_single_path_qc = zeros(size(full_single_path));
            full_single_path_qc(save_linear_indices{1}) = 1;
            full_single_path_qc = imdilate(full_single_path_qc, strel('disk', 2));
            full_single_path_qc(imbinarize(full_single_path)) = 1;
            
            mip = max(full_single_path_qc, [], 3); mip = im2double(mip);
            imwrite(mip, strcat(filename_save,'_originalIdx_', int2str(originalIdx), '_RECURSIVE_MAX.tif') , 'writemode', 'append', 'Compression','none')
        end
        
        %% (3) determine if last index of originalIdx is a BRANCHPOINT
        % only true if at least >= 2 branches stemming from original
        %         % segment
        %         if length(branchIdx) >= 2
        %             branchpoint = save_linear_indices{1}(end);
        %             full_single_path(branchpoint) = 2;
        %         end
        
        
        %% Don't need to generate branch-points below, b/c already know where they are
        % val 1 == normal skeleton
        % val 2 == branchpoint (or overlap point)
        % val 3 == paranode   ==> will need to class weight these last 2
        % for training!!!
        
        
        %% SCALE the crops to target size!!! By first finding out how big the crop should be to be equivalent
        %% later will use "imresize" to scale to target_crop_size
        target_crop_size = 80;
        scaled_crop_size = round(target_crop_size/scale_factor);
        crop_size = scaled_crop_size/2;
        
        z_size = 16;
        seed_every = 10;
        %for path_idx = 1:length(save_linear_indices)
        path_indices = save_linear_indices{1};
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
            
            %figure(1); volshow(crop_input, 'alphamap', linspace(0,0.3,256)');
            plot_crop_tmp = full_single_path;
            plot_crop_tmp(seed_mid_point) = 5;
            plot_crop_tmp(seed_front_seg) = 4;
            plot_crop_tmp(seed_back_seg) = 6;
            crop_combined = crop_around_centroid(plot_crop_tmp, y, x, z, crop_size, z_size, height, width, depth);
            
            crop_full_path = crop_around_centroid(full_single_path, y, x, z, crop_size, z_size, height, width, depth);


            
            %% March 1st: added - seed should be FRONT SEG. BACK SEG should be part of TRUTH data!!!
            crop_combined(crop_combined == 6) = 1;
            crop_full_path(crop_full_path == 6) = 1;
 
            
            %% NEW - March 1st: skip if only have less than 4 pixels of truth data
            if length(find(crop_combined == 1)) <= 2
                continue;
            end
            
            
            %% NEW - March 1st: also, should add all elements from recurse_tree_backwards to seed!!!
            
            % if want to find those, just index with val == 7
            crop_combined(crop_combined == 7) = 4;
            crop_full_path(crop_full_path == 7) = 4;
            
            
            %% Eliminate objects in the crop that are NOT connected with the start OR end segments!!!
            new_crop = imbinarize(crop_combined);
            cc = bwconncomp(new_crop);
            regions = regionprops3(cc, 'VoxelIdxList');
            
            for region_idx = 1:length(regions.VoxelIdxList)
                voxel_ind = regions.VoxelIdxList{region_idx};
                max(crop_combined(voxel_ind));
                if max(crop_combined(voxel_ind)) < 4   % b/c pixel val 3 == paranode
                    new_crop(voxel_ind) = 0;
                end
            end
            crop_combined(new_crop == 0) = 0;  % mask original crop
            crop_full_path(new_crop == 0) = 0;  % mask full path crop as well
            %figure(2); volshow(crop_combined); title('Not cleaned');
            %figure(3); volshow(crop_full_path); title('Not cleaned');
            
            
            %% NEW - March 1st - also eliminate side branches (attached to seed but starting BEHIND
            %% MIDPOINT). Do this by (1) subtract out seed. (2) Only keeping everything that colocalizes
            %% with a dilated midpoint. (3) then add seeds back in.
            %% Hope that this reduces number of side-branch seg errors
            
            % (1) Subtract out seed
            new_crop_no_seed = imbinarize(crop_combined);
            new_crop_no_seed(crop_combined > 1) = 0;
            %new_crop_no_seed(crop_combined == 7) = 0;
            
            % (2) Dilate midpoint and then find/keep everything that is
            % colcalized with it ONLY
            mid_point = zeros(size(crop_combined));
            mid_point(round(crop_size) + 1, round(crop_size) + 1, round(z_size/2)) = 1;
            mid_point = imdilate(mid_point, strel('sphere', 3));
            
            % find colocalized
            combined = new_crop_no_seed + mid_point;
            cc = bwconncomp(new_crop_no_seed);
            regions = regionprops3(cc, 'VoxelIdxList');
            
            for region_idx = 1:length(regions.VoxelIdxList)
                voxel_ind = regions.VoxelIdxList{region_idx};
                max(combined(voxel_ind));
                if max(combined(voxel_ind)) < 2   % b/c pixel val 2 is max of 2 sums
                    combined(voxel_ind) = 0;
                end
            end
            
            save_seeds = crop_combined;
            
            crop_combined(combined == 0) = 0;  % mask original crop
            crop_full_path(combined == 0) = 0;  % mask full path crop as well
            %figure(2); volshow(crop_combined); title('Not cleaned');
            %figure(3); volshow(crop_full_path); title('Not cleaned');
            
            % (3) Restore seeds
            crop_combined(save_seeds == 4) = 4;
            crop_combined(save_seeds == 5) = 5;
            crop_combined(save_seeds == 7) = 7;            
            
            
            %% Create seed:
            seed_2 = zeros(size(crop_combined));
            seed_2(crop_combined == 4) = 1;
            % ELIMINATE ANY SEEDS THAT ARE TOO SMALL
            skip_seed_2 = 0;
            if length(find(seed_2 == 1)) < 4
                skip_seed_2 = 1;
            end
            seed_2 = imbinarize(seed_2);
            %figure(4); volshow(im2double(seed_2)); title('Cleaned by branchpoint');
            
            
            
            %% Make truth images:
            % class 1 ==> cell segments
            % class 2 ==> branchpoints
            % class 3 ==> paranodes
            truth_2_class_1 = imbinarize(crop_combined);
            truth_2_class_1(seed_2 == 1) = 0;
            %figure(5); volshow(im2double(truth_2_class_1)); title('Cleaned by branchpoint');

            
            %% class 2 == branchpoints
            %truth_2_class_2 = truth_2_class_1;
            %truth_2_class_2(crop_full_path ~= 2) = 0;
            %truth_2_class_2_dil = imdilate(truth_2_class_2, strel('sphere', 4));
            %figure(6); volshow(im2double(truth_2_class_2_dil)); title('Cleaned by branchpoint');
            
            %% class 3 == paranodes
            truth_2_class_3 = truth_2_class_1;
            truth_2_class_3(crop_full_path ~= 3) = 0;
            %figure(7); volshow(im2double(truth_2_class_3_dil)); title('Cleaned by branchpoint');
            
                       
            
             crop_input = imresize3(crop_input, [target_crop_size, target_crop_size, z_size], 'method', 'cubic');
             seed_2 = imresize3(im2double(seed_2), [target_crop_size, target_crop_size, z_size], 'method', 'cubic');
             truth_2_class_1 = imresize3(im2double(truth_2_class_1), [target_crop_size, target_crop_size, z_size], 'method', 'cubic');
             truth_2_class_3 = imresize3(im2double(truth_2_class_3), [target_crop_size, target_crop_size, z_size], 'method', 'cubic');

             seed_2 = imbinarize(seed_2);
             truth_2_class_1 = imbinarize(truth_2_class_1);
             truth_2_class_3 = imbinarize(truth_2_class_3);
             
            
            %% ERROR: cannot skeletonize on ANYTHING after dilation because totally fucks it
            %% need to erode first to restore after dilation
            %seed_2 = imerode(seed_2, strel('sphere', 1));
            %truth_2_class_1 = imerode(truth_2_class_1, strel('sphere', 1));
            %truth_2_class_3 = imerode(truth_2_class_3, strel('sphere', 1));

            %if scale_factor < 1
                seed_2 = bwskel(seed_2);
                truth_2_class_1 = bwskel(truth_2_class_1);
                truth_2_class_3 = bwskel(truth_2_class_3);
            %end
            
            
            %% Know if paranodes exist
            paranodes_exist = 0;
            if length(unique(truth_2_class_3)) > 1
                paranodes_exist = 1;
            end
            
            %% Scale and THEN dilate
            seed_2_dil = imdilate(seed_2, strel('sphere', 1));
            truth_2_class_1_dil = imdilate(truth_2_class_1, strel('sphere', 1));            
            truth_2_class_3_dil = imdilate(truth_2_class_3, strel('sphere', 2));
            

            %% Make sure seed doesn't screw with truth image
            truth_2_class_1_dil(seed_2_dil > 0) = 0; 
            
             
            %plot_max(crop_input);
            %plot_max(seed_2);
            %plot_max(truth_2_class_1);
            %plot_max(truth_2_class_3);
            
            %% Plot quality control
            cd(foldername)
            cd('./mip quality check');
            if mod(qc_counter, 1) == 0
                mip = max(crop_input, [], 3); mip = im2double(mip);
                imwrite(mip, strcat(filename_save,'_originalIdx_', int2str(originalIdx), '_seed_end_', int2str(small_seed_idx), '_NOCLAHE_input_crop_MAX.tif') , 'writemode', 'append', 'Compression','none')
                
                mip = max(seed_2_dil, [], 3); mip = im2double(mip);
                imwrite(mip, strcat(filename_save,'_originalIdx_', int2str(originalIdx), '_seed_end_', int2str(small_seed_idx), '_DILATE_seed_crop_MAX.tif') , 'writemode', 'append', 'Compression','none')
                
                mip = max(truth_2_class_1_dil, [], 3);mip = im2double(mip);
                imwrite(mip, strcat(filename_save,'_originalIdx_', int2str(originalIdx), '_seed_end_', int2str(small_seed_idx), '_DILATE_truth_class_1_crop_MAX.tif') , 'writemode', 'append', 'Compression','none')
                
                mip = max(truth_2_class_3_dil, [], 3); mip = im2double(mip);
                imwrite(mip, strcat(filename_save,'_originalIdx_', int2str(originalIdx), '_seed_end_', int2str(small_seed_idx), '_DILATE_truth_class_3_crop_MAX.tif') , 'writemode', 'append', 'Compression','none')
                
            end
            qc_counter = qc_counter + 1;

            %% Save training data
            %crop=crop-min(crop(:)); % shift data such that the smallest element of A is 0
            %crop=crop/max(crop(:)); % normalize the shifted data to 1
            cd(cur_dir);
            cd(foldername);
            cd('./TRAINING FORWARD PROP ONLY SCALED');
            
            %figure(1); volshow(crop_input);
            for k = 1:z_size
                if ~skip_seed_2
                    input = crop_input(:, :, k);
                    
                    %% save input images
                    imwrite(input, strcat(filename_save,'_originalIdx_', int2str(originalIdx), '_seed_end_', int2str(small_seed_idx), '_NOCLAHE_input_crop.tif') , 'writemode', 'append', 'Compression','none')
                    % save cropped seeds
                    input = seed_2_dil(:, :, k);
                    input = im2double(input);
                    imwrite(input, strcat(filename_save,'_originalIdx_', int2str(originalIdx), '_seed_end_', int2str(small_seed_idx), '_DILATE_seed_crop.tif') , 'writemode', 'append', 'Compression','none')
                    
                    % save truth images
                    input = truth_2_class_1_dil(:, :, k);
                    input = im2double(input);
                    imwrite(input, strcat(filename_save,'_originalIdx_', int2str(originalIdx), '_seed_end_', int2str(small_seed_idx), '_DILATE_truth_class_1_crop.tif') , 'writemode', 'append', 'Compression','none')
                    
                    %input = truth_2_class_2_dil(:, :, k);
                    %imwrite(input, strcat(filename_save,'_originalIdx_', int2str(originalIdx), '_seed_end_', int2str(small_seed_idx), '_DILATE_truth_class_2_crop.tif') , 'writemode', 'append', 'Compression','none')
                    
                    input = truth_2_class_3_dil(:, :, k);
                    input = im2double(input);
                    
                    if paranodes_exist == 0
                        imwrite(input, strcat(filename_save,'_originalIdx_', int2str(originalIdx), '_seed_end_', int2str(small_seed_idx), '_DILATE_truth_class_3_crop_neg.tif') , 'writemode', 'append', 'Compression','none')
                    else
                        imwrite(input, strcat(filename_save,'_originalIdx_', int2str(originalIdx), '_seed_end_', int2str(small_seed_idx), '_DILATE_truth_class_3_crop_pos.tif') , 'writemode', 'append', 'Compression','none')
                    end
                    
                end
            end
            cd(cur_dir)
            skip_seed_2
            %close all;
            
        end
        %% (4) Augment indices
        originalIdx = originalIdx + 1;
        
    end
end










