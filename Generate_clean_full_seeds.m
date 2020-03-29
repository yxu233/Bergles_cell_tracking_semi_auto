%% Resize all images and create clean starting seeds with up to first 2 branches? if small
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
    %originalIdx = 1;   % to start things off, first index is always zero
    qc_counter = 0;
    
    idx_original_seed = find([sortedS(:).startsOn] == 0);
    
    all_original_seeds = zeros([height, width, depth]);
    all_original_seeds_size_limited = zeros([height, width, depth]);
    for loop_idx = 1:length(idx_original_seed)
        
        %% Always plot current index + all branchs that "start on" current index
        
        all_indices = [];
        save_linear_indices = cell(0);
        
        %% (1) Plots current index segment
        x = sortedS(loop_idx).points.x;
        y = sortedS(loop_idx).points.y;
        z = sortedS(loop_idx).points.z;
        % EVERYTHING NEEDS TO BE + 1 because imageJ coords start from 0
        x = x + 1;
        y = y + 1;
        z = z + 1;
        linear_ind = sub2ind(size(input_im), y, x, z);
        all_indices = [all_indices; linear_ind];
        save_linear_indices{end + 1} = linear_ind;
        all_original_seeds(linear_ind) = 1;
        
        
        originalIdx = sortedS(loop_idx).originalIdx;  %loop through properly!!!
        
        
        %plot_max(all_original_seeds);
        %% (2) keep getting subsequent branches if current one is too short for initial seed
        length(all_indices)
        depth_limit = 10000;
        [all_original_seeds, save_linear_indices] = recurse_branch_tree(originalIdx, all_original_seeds, sortedS, depth_limit, size(input_im), save_linear_indices);
        
        %% Only keep up to 50 pixels in length
        %all_indices = all_indices(1: 30)
        all_indices = [];
        for ind_idx = 1:length(save_linear_indices)
            cur_segment = save_linear_indices{ind_idx};
            all_indices = [all_indices; cur_segment];
        end
        all_original_seeds_size_limited(all_indices) = loop_idx;  % give each seed a unique value
        
        
    end
    
    
    %% If you are UPSAMPLING, must dilate first for upsampling, and then skeletonize again after
    if scale_factor < 1
        all_original_seeds_size_limited = imdilate(all_original_seeds_size_limited, strel('sphere', 2));
    end
    
    %% Scale everything to target_size
    all_original_seeds_size_limited = imresize(all_original_seeds_size_limited,scale_factor, 'method', 'nearest');
    
    input_im = imresize(input_im,scale_factor, 'method', 'bicubic');
    
    %% Then skeletonize again at the end by using a bw skel mask
    all_original_seeds_bw = bwskel(imbinarize(all_original_seeds_size_limited));
    
    all_original_seeds_size_limited(all_original_seeds_bw == 0) = 0;
    %all_original_seeds = imbinarize(all_original_seeds);
    %all_original_seeds = imdilate(all_original_seeds, strel('sphere', 2));
    
    
    %% Plot quality control and seeds
    cd(foldername)
    cd('./seed generation individual branches');
    mip = max(all_original_seeds_size_limited, [], 3); mip = im2double(mip);
    imwrite(mip, strcat(filename_save, '_seed_end_', int2str(fileNum), '_MAX.tif') , 'writemode', 'append', 'Compression','none')
    
    mip = max(input_im, [], 3); mip = im2double(mip);
    imwrite(mip, strcat(filename_save, '_seed_end_', int2str(fileNum), '_input_MAX.tif') , 'writemode', 'append', 'Compression','none')
    
    %figure(1); volshow(crop_input);
    %all_original_seeds_size_limited(all_original_seeds_size_limited > 0) = 255;
    for k = 1:length(all_original_seeds_size_limited(1,1, :))
        input = all_original_seeds_size_limited(:, :, k)/255;
        
        %% save input images
        imwrite(input, strcat(filename_save,'_seed_end_', int2str(fileNum), '_seeds.tif') , 'writemode', 'append', 'Compression','none')

        input = input_im(:, :, k);
        imwrite(input, strcat(filename_save,'_seed_end_', int2str(fileNum), '_input.tif') , 'writemode', 'append', 'Compression','none')
        
    end
    cd(cur_dir)
    
end











