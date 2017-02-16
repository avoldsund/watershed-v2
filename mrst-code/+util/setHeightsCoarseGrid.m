function z = setHeightsCoarseGrid(CG, heights, trapHeights, nrOfTraps)
%SETHEIGHTSCOARSEGRID Set the heights of a coarsened grid.
%   z = SETHEIGHTSCOARSEGRID(CG, HEIGHTS, TRAPHEIGHTS, NROFTRAPS) returns
%   the heights of the remaining cells in the grid.
%   
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
%   In the coarsened grid we get the cell ordering:
%     #  #  #  #  #  #           10 10 10 10 10 10
%     #  #  #  8  9  #           10  9  9  9  7 10
%     #  #  6  7  9  #           10  9 10  9  7 10
%     #  2  3  4  5  #            8 10 10 10  7 10
%     # 10 10 10  1  #           10  4  4 4 4.5 10
%     #  #  #  #  #  #           10  4 10 10 10 10
%    
%   Output: z = [4.5, 10, 10, 10, 7, 10, 9, 9, 7, 4]
%   Notice that the traps are combined in CG, and their heights are given
%   last in the z-vector.

cutoff = CG.cells.num - nrOfTraps;
notTrapCells = CG.parent.cells.indexMap(CG.partition <= cutoff);

z = zeros(CG.cells.num, 1);
z(1:size(notTrapCells, 1)) = heights(notTrapCells);
z(size(notTrapCells, 1) + 1:end) = trapHeights;

end