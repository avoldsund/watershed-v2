% Load necessary data and compute geometry
figure()
load('watershed.mat');
load('heights.mat');
[nRows, nCols] = size(heights);
heights = heights';
heights = rot90(heights, -1);
ws = util.mapCoordsToIndices(watershed', nCols, nRows);
cellsInWatershed = size(watershed, 2);

traps = load('traps.mat');
traps = traps.traps;

% Create nRows x nCols-grid and remove some cells
G = cartGrid([nCols, nRows]);
G = computeGeometry(G);
G.cells.z = heights(1:nRows * nCols);

% Remove all cells in the grid not in the watershed
rmCells = setdiff(1 : nCols * nRows, ws);
G = removeCells(G, rmCells);

Z = reshape(G.cells.z, [nRows, nCols]);
clf, mesh(Z)
xlabel('x');
ylabel('y');

% newplot
% plotGrid(G,'FaceColor',[0.95 0.95 0.95]); axis off;
z = ones(1,nRows*nCols);
z(ws) = heights(ws);
figure()
mesh(reshape(z, [nRows, nCols])')