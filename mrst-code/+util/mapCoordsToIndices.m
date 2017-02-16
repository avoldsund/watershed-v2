function indices = mapCoordsToIndices(coords, nCols, nRows)
% MAPCOORDSTOINDICES Map coordinates to cell indices.
%   indices = MAPCOORDSTOINDICES(COORDS, NCOLS, NROWS) first maps the 
%   coordinates from a 0-indexed matrix to a 1-indexed matrix, and then 
%   maps the coords to indices where the 1st index is at the bottom left.
%  
%   Examples:
%      Ex 1:       
%       1. Map coords to 1-based indexing
%       coords = [0, 1; 1, 0; 2, 2;]
%       nCols = 3
%       nRows = 3
%       
%       (0,0) (0,1) (0,2)    (1,1) (1,2) (1,3)
%       (1,0) (1,1) (1,2) -> (2,1) (2,2) (2,3)
%       (2,0) (2,1) (2,2)    (3,1) (3,2) (3,3)
%       
%       newCoords = [1, 2; 2, 1; 3, 3;]
% 
%       2. Map new coords to cell indices where they are ordered
%       7 8 9
%       4 5 6
%       1 2 3
% 
%       cellIndices -> [8, 4, 3]
%
%      Ex 2:
%       coords = [0, 1; 0, 3; 1, 0; 2, 0; 2, 2;]
%       nCols = 4
%       nRows = 3
%
%       9 10 11 12
%       5  6  7  8
%       1  2  3  4
%       
%       cellIndices -> [10, 12, 5, 1, 3]

coords = coords + 1;
indices = (nRows - coords(:, 1)) * nCols + coords(:, 2);

end