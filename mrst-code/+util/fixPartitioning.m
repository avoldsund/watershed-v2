function [partition, spillPoints] = fixPartitioning(G, traps, nrOfTraps, spillPairs)
% FIXPARTITIONING Creates the partitioning necessary to coarsen the grid.
%   partition = FIXPARTITIONING(G, TRAPS, NROFTRAPS) returns a partitioning
%   using the grid structure, the traps and the nr of traps.
%
%   partition represents how cells are combined. The output shows that the
%   cells 8, 9 and 10 shall be combined into cell nr 10, whereas e.g., cell
%   nr 11 becomes cell 1, and as there is no other 1's in the partition
%   vector, it will not be combined with any other cells.
%
% Example:
%   Ex 1:
%    How the watershed looks:    Heights:
%     #  #  #  #  #  #           10 10 10 10 10 10
%     #  #  # 28 29  #           10  9  9  9  7 10
%     #  # 21 22 23  #           10  9 10  9  7 10
%     # 14 15 16 17  #            8 10 10 10  7 10
%     #  8  9 10 11  #           10  4  4 4 4.5 10
%     #  #  #  #  #  #           10  4 10 10 10 10
%   
%    Traps: 
%    1. [29, 23]
%    2. [8, 9, 10]
%   
%   cells = [8, 9, 10, 11, 14, 15, 16, 17, 21, 22, 23, 28, 29]
%   Output: partition = [10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 9, 8, 9]'

nCols = G.cartDims(1);
nRows = G.cartDims(2);
traps = util.mapListOfCoordsToIndices(traps, nCols, nRows);
spillPairs = util.mapListOfCoordsToIndices(spillPairs, nCols, nRows);
spillPoints = zeros(size(spillPairs, 1), 1);

totalCells = G.cartDims(1) * G.cartDims(2);
map = zeros(totalCells, 1);
map(G.cells.indexMap) = 1:G.cells.num;
partition = 1:G.cells.num;

for i = 1:nrOfTraps
    tCells = traps{i};
    spillPoint = spillPairs(i);
    trapIndices = map(tCells);
    spillPoints(i) = map(spillPoint);
    partition(trapIndices) = G.cells.num + i;
end

partition = compressPartition(partition);

end

