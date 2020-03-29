function [full_single_path, save_linear_indices] = recurse_branch_tree_by_pixels(originalIdx, full_single_path, sortedS, depth_limit, siz, save_linear_indices)

    all_indices = [];
    for ind_idx = 1:length(save_linear_indices)
        cur_segment = save_linear_indices{ind_idx};
        all_indices = [all_indices; cur_segment];
    end

    % breaks when reaches pixel limit
    if length(all_indices) > depth_limit
        depth_limit = 0;
    
    else
        branchIdx = find([sortedS.startsOn] == originalIdx);
        if ~isempty(branchIdx)
            for id = 1:length(branchIdx)
                b_idx = branchIdx(id);
                branchIdxVal = sortedS(b_idx).originalIdx;
                x = sortedS(b_idx).points.x;
                y = sortedS(b_idx).points.y;
                z = sortedS(b_idx).points.z;
                % EVERYTHING NEEDS TO BE + 1 because imageJ coords start from 0
                x = x + 1;
                y = y + 1;
                z = z + 1;
                linear_ind = sub2ind(siz, y, x, z);
                %all_indices = [all_indices; linear_ind];
                save_linear_indices{end + 1} = linear_ind;
                full_single_path(linear_ind) = 1;

                % Determine if the tips of this segment contains a
                % branchpoint ==> true if >= 2 branchs stemming from original
%                 nextBranchIdx = find([sortedS.startsOn] == branchIdxVal);
%                 if length(nextBranchIdx) >= 2
%                     branchpoint = linear_ind(end);
%                     full_single_path(branchpoint) = 2;
%                 end

                % Also check for paranodes
                name = sortedS(b_idx).attribs.name;
                if contains(name, 's')
                    paranode_idx = linear_ind(end);
                    full_single_path(paranode_idx) = 3;
                end
                [full_single_path, save_linear_indices] = recurse_branch_tree_by_pixels(branchIdxVal, full_single_path, sortedS, depth_limit - 1, siz, save_linear_indices);
                
            end
        end   
    end
    

end