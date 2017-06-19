% Load necessary data and compute geometry
figure()
l = load('/+landscapes/landscapeTyrifjordenWatershed.mat');
watershed = l.watershed; 
outlet = l.outlet; 
faceLength = double(l.stepSize);
heights = l.heights;
traps = l.traps;
nrOfTraps = l.nrOfTraps;
trapHeights = l.trapHeights;
flowDirections = l.flowDirections;
spillPairs = l.spillPairs;
[nRows, nCols] = size(heights);

heights = rot90(heights, -1);
ws = util.mapCoordsToIndices(watershed', nCols, nRows);
cellsInWatershed = size(watershed, 2);

% Create nRows x nCols-grid and remove some cells
G = cartGrid([nCols, nRows]);
G = computeGeometry(G);
G.cells.z = heights(1:nRows * nCols);

% Remove all cells in the grid not in the watershed
rmCells = setdiff(1 : nCols * nRows, ws);
G = removeCells(G, rmCells);

%Z = reshape(G.cells.z, [nRows, nCols]);
%clf, mesh(Z)
%xlabel('x');
%ylabel('y');

% newplot
% plotGrid(G,'FaceColor',[0.95 0.95 0.95]); axis off;
z = NaN(1,nRows*nCols);
z(ws) = heights(ws);
figure()
hold on
mesh(reshape(z, [nRows, nCols])')


z = NaN(1,nRows*nCols);
c = find(G.cells.z == 63);
z(c) = 63;
surf(reshape(z, [nRows, nCols])')