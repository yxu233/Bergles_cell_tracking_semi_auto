function [full_single_path, save_linear_indices] = recurse_branch_tree_backwards(originalIdx, full_single_path, sortedS, depth_limit, siz, save_linear_indices)

    if depth_limit == 0
        depth_limit = 0;
    else
        branchIdx = find([sortedS.originalIdx] == originalIdx);
        if ~isempty(branchIdx)
            for id = 1:length(branchIdx)
                b_idx = branchIdx(id);
                parentIdxVal = sortedS(b_idx).startsOn;
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
                full_single_path(linear_ind) = 7;

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
                [full_single_path, save_linear_indices] = recurse_branch_tree_backwards(parentIdxVal, full_single_path, sortedS, depth_limit - 1, siz, save_linear_indices);
                
            end
        end   
    end
    

end