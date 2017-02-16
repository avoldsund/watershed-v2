function listOfIndices = mapListOfCoordsToIndices(listOfCoords, nCols, nRows)
%MAPLISTOFCOORDSTOINDICES Maps a Nx2-cell array with (row, column)
% coordinates to a Nx1-cell array with indices (origin lower left and 1-indexed).
%
%   listOfIndices = MAPLISTOFCOORDSTOINDICES(LISTOFCOORDS, NCOLS, NROWS) first maps the 
%   coordinates from a 0-indexed matrix to a 1-indexed matrix, and then 
%   maps the coords to indices where the 1st index is at the bottom left.
%   It will map every row of a Nx2-cell array such that the output is a
%   Nx1-cell array.
%
%   Examples:
%      Ex 1:       
%       listOfCoords = cell(2,2);
%       listOfCoords{1,1} = [1, 2];
%       listOfCoords{1,2} = [4, 4];
%       listOfCoords{2,1} = [4, 4, 4];
%       listOfCoords{2,2} = [1, 2, 3];
%       
%       nCols = 6; nRows = 6;
%
%       listOfIndices = cell(2,1);
%       listOfIndices{1,1} = [29, 23];
%       listOfIndices{2,1} = [8, 9, 10];
%       
%       return listOfIndices

    N = size(listOfCoords, 1);
    listOfIndices = cell(N, 1);
    
    for i = 1:N
        coords = horzcat(listOfCoords{i, 1}', listOfCoords{i, 2}');
        listOfIndices{i} = util.mapCoordsToIndices(coords, nCols, nRows);
    end
    
end

