function flowDirectionArray = getFlowDirections(CG, flowDirections, nrOfTraps)
% GETFLOWDIRECTIONS Set flow directions of coarse grid with traps
%   FLOWDIR = GETFLOWDIRECTIONS(CG, FLOWDIRECTIONS, NROFTRAPS) returns the
%   flow direction vectors for all cells in the grid. The trap cells have a
%   flow direction of [0, 0].
%   
%   How the watershed looks:       Flow directions:
%    #  #  #  #  #  #              0  0  0  0  0  0
%    #  #  #  8 |9| #              0 -1 -1  2 -1  0
%    #  #  6  7 | | #              0 16  2  2 -1  0
%    #  2  3  4  5  #              0  8  8  8  8  0
%    #  | 10  |  1  #              0 -1 -1 -1 32  0
%    #  #  #  #  #  #              0  0  0  0  0  0
%
%   Notice that the watershed pictured shows the indexing of the cells in
%   the coarse grid (CG). The two traps have been combined into two single 
%   cells.
%   
%   flowDir = [32, 8, 8, 8, 8, 2, 2, 2, -1, -1]'
%
%   flowDir is then mapped to a vector using the mapping below
%             x  y
%       1 ->  1  1
%       2 ->  1  0
%       4 ->  1 -1
%       8 ->  0 -1
%      16 -> -1 -1
%      32 -> -1  0
%      64 -> -1  1 
%     128 ->  0  1
%
%   Output: [-1, 0; 0, -1; 0, -1; 0, -1; 0, -1; 1, 0; 1, 0; 1, 0; 0, 0; 0, 0;]

% Set flow directions
cutoff = CG.cells.num - nrOfTraps;
notTrapCells = CG.parent.cells.indexMap(CG.partition <= cutoff);
flowDir = zeros(CG.cells.num, 1);
flowDir(1:size(notTrapCells, 1)) = flowDirections(notTrapCells);
flowDir(size(notTrapCells, 1) + 1:end) = -1;

% Map to flow direction vectors
N = CG.cells.num - nrOfTraps;
flowDirectionArray = zeros(CG.cells.num, 2);
mapTo = [1, 1; 1, 0; 1, -1; 0, -1; -1, -1; -1, 0; -1, 1; 0, 1;];
flowDirectionArray(1:N, :) = mapTo(log2(flowDir(1:N)) + 1, :);

end