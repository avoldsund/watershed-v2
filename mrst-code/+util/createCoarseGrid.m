function CG = createCoarseGrid(watershed, heights, traps, nrOfTraps, spillPairs, stepSize)
%CREATECOARSEGRID Creates a coarse grid based on watershed and traps. 
%   CG = CREATECOARSEGRID(WATERSHED, HEIGHTS, TRAPS, NROFTRAPS, SPILLPAIRS, STEPSIZE)
%   first creates a cartGrid the same size as the HEIGHTS-landscape. Each
%   cell is stepSize x stepSize large. Some cells are removed, so only the
%   WATERSHED cells remain. Afterwards TRAPS are combined in a process using
%   the original grid G, TRAPS, NROFTRAPS and SPILLPAIRS. The coarsened
%   grid CG is returned.

[nRows, nCols] = size(heights);

% Create entire nRows x nCols-grid and remove cells outside watershed
%stepSize = 10;
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

