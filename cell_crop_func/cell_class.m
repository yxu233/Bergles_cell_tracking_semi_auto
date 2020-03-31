classdef cell_class
    properties
        voxelIdxList
        centroid
        cell_number
    end
    methods
        %% constructor functions
        function obj = cell_class(voxelIdxList, centroid, cell_number)
            if nargin == 3
                obj.voxelIdxList = voxelIdxList;
                obj.centroid = centroid;
                obj.cell_number = cell_number;
            end
        end
    end
end