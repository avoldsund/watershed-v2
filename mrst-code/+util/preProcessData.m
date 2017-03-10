function [heights, fd, ws, spillPairsIndices] = preProcessData(heights, flowDirections, watershed, spillPairs)
%PREPROCESSDATA Summary of this function goes here
%   Detailed explanation goes here

[nRows, nCols] = size(heights);
heights = rot90(heights, -1);  % Fix 1d-indexing
fd = rot90(flowDirections, -1);  % Fix 1d-indexing
ws = util.mapCoordsToIndices(watershed', nCols, nRows);
spillPairsIndices = util.mapListOfCoordsToIndices(spillPairs, nCols, nRows);

end

