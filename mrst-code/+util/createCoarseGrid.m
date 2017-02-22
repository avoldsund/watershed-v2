function CG = createCoarseGrid(watershed, heights, traps, nrOfTraps, spillPairs)
%CREATECOARSEGRID Summary of this function goes here
%   Detailed explanation goes here

[nRows, nCols] = size(heights);

% Create entire nRows x nCols-grid and remove cells outside watershed
stepSize = 10;
G = cartGrid([nCols, nRows], [nCols * stepSize, nRows * stepSize]);
G = computeGeometry(G);
rmCells = setdiff(1 : nCols * nRows, watershed);
G = removeCells(G, rmCells);

% Combine traps and coarsen gridnrOfTraps
[partition, spillPoints] = util.fixPartitioning(G, traps, nrOfTraps, spillPairs);
CG = generateCoarseGrid(G, partition);
CG = coarsenGeometry(CG);
CG.spillPoints = spillPoints;

end

