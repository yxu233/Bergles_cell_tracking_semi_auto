function [option_num, matrix_timeseries] = Bergles_manual_correct(frame_1, frame_2, truth_1, truth_2, crop_frame_2, D, check_neighbor, neighbor_idx, matrix_timeseries, cur_timeseries, next_timeseries, timeframe_idx, x_min, x_max, y_min, y_max, z_min, z_max, crop_size, z_size,cur_centroids, next_centroids, dist_thresh, ssim_val_thresh)

% Allows user to manually correct the counted the fully counted image:

term = 0;
% Plot
[ssim_val, dist] = plot_im(0);


%% 2nd check of ssim and distance metric with expanded crop
if ssim_val > ssim_val_thresh && dist < dist_thresh
    option_num = 10;
else
    option_num = 1;
end

sorted_idx = 0;
option_new_cell = 0;
%% Now add selection menu


while option_num>0 && term == 0
    
    try
        if option_num == 10
            disp('skip and add')
            option_num = 1;
        else
            k = getkey(1,'non-ascii');
            option_num=str2double(k);
            if isnan(option_num)
                option_num = k;
            end
        end
        %% If key == 1, then is correctly matched cell
        if option_num==1
            if option_new_cell == 0
                next_cell = next_timeseries(neighbor_idx(check_neighbor));
                voxelIdxList = next_cell.objDAPI;
                centroid = next_cell.centerDAPI;
                cell_num = check_neighbor;
                % create cell object
                cell_obj = cell_class(voxelIdxList,centroid, cell_num);
                matrix_timeseries{check_neighbor, timeframe_idx + 1} = cell_obj;
                
                %% add newly selected cell from option 3 below
            elseif option_new_cell == 1
                disp('yeet')
                next_cell = next_timeseries(sorted_idx);
                voxelIdxList = next_cell.objDAPI;
                centroid = next_cell.centerDAPI;
                cell_num = check_neighbor;
                % create cell object
                cell_obj = cell_class(voxelIdxList,centroid, cell_num);
                matrix_timeseries{check_neighbor, timeframe_idx + 1} =  cell_obj;
                
                sorted_idx = 0;
                option_new_cell = 0;
            end
            
            
            term = 1;
            break;
            
            %% If key == 2, then NOT the same cell, so don't include
        elseif option_num==2
            term = 2;
            
            %% If key == a, "add" then need to assign to NEW point
        elseif option_num=='a'
            
            cell_point=impoint;
            % Get XY position from placed dot
            poss_sub =getPosition(cell_point);
            
            % Get z-position from prompt
            prompt = {'Enter z-axis position:'};
            dlgtitle = 'slice position';
            definput = {'11'};
            answer = inputdlg(prompt,dlgtitle, [1, 35], definput);
            
            coordinates = [round(poss_sub), str2num(answer{1})];
            
            % create blank volume with only selected point
            blank_vol = zeros(size(crop_frame_2));
            blank_vol(sub2ind(size(crop_frame_2),coordinates(2), coordinates(1), coordinates(3))) = 1;
            
            % dilate to make matching easier
            blank_vol = imdilate(blank_vol, strel('sphere', 2));
            
            
            % insert point into larger volume
            full_vol = zeros(size(truth_2));
            full_vol(x_min:x_max, y_min:y_max, z_min:z_max) = blank_vol;
            
            % find index of point in relation to larger volume
            linear_idx = find(full_vol);
            
            % search through cells in next time frame to find which ones match
            matched = 0;
            for sorted_idx = 1:length(next_timeseries)
                if isempty(next_timeseries(sorted_idx).objDAPI)
                    continue;
                end
                sorted_cell = next_timeseries(sorted_idx).objDAPI;
                same = ismember(linear_idx, sorted_cell);
                
                % if matched
                if ~isempty(find(same, 1))
                    matched = 1;
                    break;
                end
            end
            
            % save corrected cell after sorting to matrix_timeseries
            if matched == 1
                option_new_cell = 1;
                disp('first yeet')
                % also, change the value in the next_centroids array (to plot updated)
                [x, y, z]= ind2sub(size(full_vol), linear_idx);
                next_centroids(neighbor_idx(check_neighbor), :) = [y(round(length(y)/2)), x(round(length(x)/2)), z(round(length(z)/2))];
                
                % if did NOT match a point, reselect
            elseif matched == 0
                f = msgbox('Points did not match existing cell, please reselect');
            end
            
            delete(cell_point);
            plot_im(0);
            
            %% If key == d, "delete" then delete current cell on current timeframe (i.e. not a real cell)
        elseif option_num == 'd'
            matrix_timeseries{check_neighbor, timeframe_idx} = [];
            term = 4;
            break;
            
            %% If key == s, "scale" then need to scale image outwards in the plot
        elseif option_num == 's'
            % Get scaling from prompt
            prompt = {'Enter desired XYZ scaling (0 - inf):'};
            dlgtitle = 'scaling requested';
            definput = {'2'};
            answer = inputdlg(prompt,dlgtitle, [1, 35], definput);
            scale_XYZ = str2num(answer{1});
            
            plot_im(scale_XYZ);
            
            %% If key == c, "clahe" then do CLAHE
        elseif option_num =='c'
            plot_im('adjust');
            
            
            %% If key == 3, add non-exisiting cell   ==> NOT YET WORKING
        elseif option_num ==3
            %3plot_im(0);
            cell_point=impoint;
            % Get XY position from placed dot
            poss_sub =getPosition(cell_point);
            
            % Get z-position from prompt
            prompt = {'Enter z-axis position:'};
            dlgtitle = 'slice position';
            definput = {'10'};
            answer = inputdlg(prompt,dlgtitle, [1, 35], definput);
            
            coordinates = [round(poss_sub), str2num(answer{1})];
            
            % create blank volume with only selected point
            blank_vol = zeros(size(crop_frame_2));
            blank_vol(sub2ind(size(crop_frame_2),coordinates(2), coordinates(1), coordinates(3))) = 1;
            
            % dilate to make matching easier
            blank_vol = imdilate(blank_vol, strel('sphere', 2));
            
            % insert point into larger volume
            full_vol = zeros(size(truth_2));
            full_vol(x_min:x_max, y_min:y_max, z_min:z_max) = blank_vol;
            
            % find index of point in relation to larger volume
            linear_idx = find(full_vol);
            
            
            %% Add dilated cell point into the matrix_timeseries!
            %next_cell = next_timeseries(neighbor_idx(check_neighbor));
            voxelIdxList = linear_idx;
            [x, y, z]= ind2sub(size(full_vol), linear_idx);
            centroid = [y(round(length(y)/2)), x(round(length(x)/2)), z(round(length(z)/2))];
            cell_num = check_neighbor;
            % create cell object
            cell_obj = cell_class(voxelIdxList,centroid, cell_num);
            matrix_timeseries{check_neighbor, timeframe_idx + 1} = cell_obj;
            
            
            %% set point as this new one in the points array for plotting
            next_centroids(neighbor_idx(check_neighbor), :) = [y(round(length(y)/2)), x(round(length(x)/2)), z(round(length(z)/2))];
            
            
            %% Plot how it looks now
            delete(cell_point);
            plot_im(0);
            
            %% Satisfied?
            %prompt = {'New point looks okay? (Y/N):'};
            %dlgtitle = 'Evaluate';
            %definput = {'Y'};
            %answer = inputdlg(prompt,dlgtitle, [1, 35], definput);
            
            %looks_okay = answer{1};
            
            %if looks_okay == 'Y'
            % option_num = 1;
            %else
            %    disp('try picking again')
            %end
            
            
            %% If key == h, then hide overlay for ease of view
        elseif option_num=='h'
            plot_im(8);
            
        else
            waitfor(msgbox('Key did not match any command, please reselect'));
            plot_im(0);
            option_num = 100;
            %pause;
            
        end
    catch
        continue;
    end
    
end


%% TO PLOT the wholeimage
    function [ssim_val, dist] = plot_im(opt)
        
        scale_XYZ = opt;
        if opt == 'adjust'
            disp('adjust');
        elseif opt == 8
            disp('hide');
        elseif opt == 10
            disp('skip pause')
        elseif opt > 0
            scale_XYZ = opt;
            original_size = crop_size;
            original_z = z_size;
            crop_size = crop_size * scale_XYZ;
            z_size = z_size * scale_XYZ;
        end
        
        
        %% get crops
        close all;
        [crop_frame_1, crop_frame_2, crop_truth_1, crop_truth_2, mip_1, mip_2, crop_blank_truth_1, crop_blank_truth_2] = crop_centroids(cur_centroids, next_centroids, frame_1, frame_2, truth_1, truth_2, check_neighbor, neighbor_idx, crop_size, z_size);
        
        frame_2_centroid = next_centroids(neighbor_idx(check_neighbor), :);
        y = round(frame_2_centroid(1)); x = round(frame_2_centroid(2)); z = round(frame_2_centroid(3));
        im_size = size(frame_2);
        height = im_size(1);  width = im_size(2); depth = im_size(3);
        [crop_frame_2, x_min, x_max, y_min, y_max, z_min, z_max] = crop_around_centroid(frame_2, y, x, z, crop_size, z_size, height, width, depth);
        
        
        %% accuracy metrics
        dist = D(check_neighbor);
        ssim_val = ssim(crop_frame_1, crop_frame_2);
        mae_val = meanAbsoluteError(crop_frame_1, crop_frame_2);
        psnr_val = psnr(crop_frame_1, crop_frame_2);
        
        crop_truth_1(crop_blank_truth_1 == 1) = 0;
        crop_truth_2(crop_blank_truth_2 == 1) = 0;
        
        if opt == 8
            RGB_1 = cat(4, crop_frame_1, crop_blank_truth_1, crop_blank_truth_1);
            RGB_2 = cat(4, crop_frame_2, crop_blank_truth_2, crop_blank_truth_2);
        else
            RGB_1 = cat(4, crop_frame_1, crop_truth_1, crop_blank_truth_1);
            RGB_2 = cat(4, crop_frame_2, crop_truth_2, crop_blank_truth_2);
        end
        
        %f = figure('units','normalized','outerposition',[0 0 1 1])
        f = figure();
        set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
        p = uipanel();
        
        top_left = uipanel('Parent',p,  'Position',[0.05 0.5 .40 .50]);
        top_right = uipanel('Parent',p, 'Position', [.55 0.5 .40 .50]);
        bottom_left = uipanel('Parent',p,  'Position',[0.05 0 .40 .50]);
        bottom_right = uipanel('Parent',p, 'Position', [.55 0 .40 .50]);
        
        s1 = sliceViewer(RGB_1, 'parent', top_left);
        s2 = sliceViewer(RGB_2, 'parent', top_right);
        
        % plot max
        if opt == 'adjust'
            mip_1 = adapthisteq(mip_1);
        end
        ax = axes('parent', bottom_left);
        imshow(mip_1);
        %image(ax, im2uint8(mip_1));
        colormap('gray'); axis off
        
        % add overlay
        if opt == 8
            disp('hide');
        else
            mip_center_1 = max(crop_blank_truth_1, [], 3);
            magenta = cat(3, ones(size(mip_1)), zeros(size(mip_1)), ones(size(mip_1)));
            hold on;
            h = imshow(magenta);
            hold off;
            set(h, 'AlphaData', mip_center_1)
        end
        
        title(strcat('ssim: ', num2str(ssim_val)))
        
        
        % plot max
        if opt == 'adjust'
            mip_2 = adapthisteq(mip_2);
        end
        ax = axes('parent', bottom_right);
        imshow(mip_2);
        colormap('gray'); axis off
        
        % add overlay
        if opt == 8
            disp('hide');
        else
            mip_center_2 = max(crop_blank_truth_2, [], 3);
            magenta = cat(3, ones(size(mip_2)), zeros(size(mip_2)), ones(size(mip_2)));
            hold on;
            h = imshow(magenta);
            hold off;
            set(h, 'AlphaData', mip_center_2)
        end
        
        title(strcat('  dist: ', num2str(dist)))
        
        % restore original crop size
        if opt == 'adjust'
            disp('adjust');
        elseif opt == 8
            disp('hide');
        elseif opt == 10
            disp('skip pause')
        elseif opt > 0
            crop_size = original_size;
            z_size = original_z;
        end
        
        
        % pause allows usage of the scroll bar
        if ssim_val > ssim_val_thresh && dist < dist_thresh
            disp('skip pause')
        else
            pause
        end
        
        
    end


end