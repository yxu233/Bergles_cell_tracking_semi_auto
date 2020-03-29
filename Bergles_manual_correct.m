function [option_num, matrix_timeseries] = Bergles_manual_correct(crop_frame_1, crop_frame_2, crop_truth_1, crop_truth_2, crop_blank_truth_1, crop_blank_truth_2,frame_1, frame_2, truth_1, truth_2, mip_1, mip_2, D, check_neighbor, neighbor_idx, matrix_timeseries, cur_timeseries, next_timeseries, timeframe_idx)

% Allows user to manually correct the counted the fully counted image:
%
% - prints on the fully counted image the outline of DAPI points using boundaries of objDAPI image
% - then allow user selection to pick DAPI points
%     1: ==> remove wrapping
%     2: ==> add wrapping
%
%     3: ==> draw fibers???
%     4: ==> also add O4+ cells???
%
% Inputs:
%         count_im = the fully counted image
%         mask = skeleton of fibers
%         s = struct containing everything
%             objDAPI ==> all the DAPI points
%
% Outputs:
%         corr_im = Corrected image
%         s ==> updated struct

%% MAYBE ADD WAY TO SHUFFLE/BLIND THE CORRECTOR???

term = 0;
% Plot
plot_im(1);

%% Now add selection menu

option_num = 1;
while option_num>0 && term == 0
    
    try
        k = getkey(1,'non-ascii');
        option_num=str2double(k);
        
        %% If key == 1, then is correctly matched cell
        if option_num==1
            next_cell = next_timeseries(neighbor_idx(check_neighbor));
            voxelIdxList = next_cell.objDAPI;
            centroid = next_cell.centerDAPI;
            cell_num = check_neighbor;
            % create cell object
            cell_obj = cell_class(voxelIdxList,centroid, cell_num);
            matrix_timeseries{check_neighbor, timeframe_idx + 1} = cell_obj;

            term = 1;
            break;
            
            %% If key == 2, then NOT the same cell, so don't include
        elseif option_num==2
            term = 2;
            
            %% If key == 3, then need to assign to NEW point
        elseif option_num==3
            
            cell_point=impoint;
            %% Get XY position from placed dot
            poss_sub =getPosition(cell_point);

            %% Get z-position from prompt
            prompt = {'Enter z-axis position:'};
            dlgtitle = 'slice position';
            answer = inputdlg(prompt,dlgtitle)

            %% Must also pass the coordinates of the cropping box to be able to put this
            %% back into the original image size, and then find the coordinates of the cell
            %% corresponding to this spot



            % Find matching DAPI point and update it:
            for i = 1:length(s)
                curDAPI = s(i).objDAPI;
                same = ismember(curDAPI, poss_idx);
                if ~isempty(find(same,1))  % Print out a * if not already wrapped
                    s(i).Core = [];
                    s(i).Bool_W = 0;   % ALSO DELETES WRAPPING b/c there's no point
                end
            end
            delete(cell_point);
            
            plot_im(1);
            
            %% If key == 4, then need to scale image outwards in the plot
        elseif option_num == 4
            %         bw = roipoly;
            %         core = bw;
            
            %% If key == 5, then do CLAHE?
        elseif option_num ==5
            plot_im(0);
            break
            
            %% If key == 6, add non-exisiting cell
        elseif option_num ==6
            plot_im(0);
            
            %% If key == 7, remove exisiting cell (by selection)
        elseif option_num == 7
            plot_im(0);
        end
    catch
        continue;
    end
    
end


%% TO PLOT the wholeimage
    function [] = plot_im(opt)
        
        %% accuracy metrics
        dist = D(check_neighbor);
        ssim_val = ssim(crop_frame_1, crop_frame_2);
        mae_val = meanAbsoluteError(crop_frame_1, crop_frame_2);
        psnr_val = psnr(crop_frame_1, crop_frame_2);
        
        crop_truth_1(crop_blank_truth_1 == 1) = 0;
        crop_truth_2(crop_blank_truth_2 == 1) = 0;
        RGB_1 = cat(4, crop_frame_1, crop_truth_1, crop_blank_truth_1);
        RGB_2 = cat(4, crop_frame_2, crop_truth_2, crop_blank_truth_2);
        
        %f = figure('units','normalized','outerposition',[0 0 1 1])
        f = figure()
        set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 0.8 0.8]);
        p = uipanel();
        
        top_left = uipanel('Parent',p,  'Position',[0.05 0.6 .40 .50]);
        top_right = uipanel('Parent',p, 'Position', [.55 0.6 .40 .50]);
        bottom_left = uipanel('Parent',p,  'Position',[0.05 0 .40 .60]);
        bottom_right = uipanel('Parent',p, 'Position', [.55 0 .40 .60]);
        
        s1 = sliceViewer(RGB_1, 'parent', top_left);
        s2 = sliceViewer(RGB_2, 'parent', top_right);
        
        ax = axes('parent', bottom_left);
        image(ax, im2uint8(mip_1));
        colormap('gray'); axis off
        title(strcat('psnr:', num2str(psnr_val),'  ssim: ', num2str(ssim_val)))
        
        ax = axes('parent', bottom_right);
        image(ax, im2uint8(mip_2));
        colormap('gray'); axis off
        title(strcat('  mae: ', num2str(mae_val), '  dist: ', num2str(dist)))
        
        if opt == 1
            disp('yay')
        end
        
    end

end

