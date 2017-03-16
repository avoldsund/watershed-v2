function [heights, fd, ws, spillPairsIndices] = preProcessData(heights, flowDirections, watershed, spillPairs)
%PREPROCESSDATA preprocesses data by doing rotations (fix 1D-indices) 
%and mapping
%   [HEIGHTS, FD, WS, SPILLPAIRSINDICES] = PREPROCESSDATA(HEIGHTS, 
%   FLOWDIRECTIONS, WATERSHED, SPILLPAIRS) preprocesses the input data and
%   returns it. This includes rotations to fix 1D-indexing, as well as
%   mapping from coordinates to indices.

[nRows, nCols] = size(heights);
heights = rot90(heights, -1);  % Fix 1d-indexing
fd = rot90(flowDirections, -1);  % Fix 1d-indexing
ws = util.mapCoordsToIndices(watershed', nCols, nRows);
spillPairsIndices = util.mapListOfCoordsToIndices(spillPairs, nCols, nRows);

end

