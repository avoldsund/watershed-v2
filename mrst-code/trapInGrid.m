%% Calculate time-of-flight using DEM with traps in landscape

% Load necessary data and compute geometry
ws = load('watershed.mat');
ws = ws.ws';
landscape = load('landscape.mat');
heights = landscape.heights;
[nRows, nCols] = size(heights);
wsIndices = util.mapCoordsToIndices(ws, nCols, nRows);

% Create nRows x nCols-grid and remove some cells
G = cartGrid([nCols, nRows]);
G = computeGeometry(G);
heights = heights';
G.cells.z = heights(wsIndices);

% Remove all cells in the grid not in the watershed
rmCells = setdiff(1 : nCols * nRows, wsIndices);
G = removeCells(G, rmCells);

%% Add flux field, state, rock and source
flux = zeros(G.faces.num, 1);
indWithFlux = all(G.faces.neighbors > 0, 2);
N = G.faces.neighbors;
flux(indWithFlux) = -(G.cells.z(N(indWithFlux, 2))- G.cells.z(N(indWithFlux, 1)));

state = struct('flux', flux);
% Make rock porosity lower on left side of grid
poro = ones(G.cells.num, 1);
ix_left = [1, 2, 6, 7, 11, 12];
poro(ix_left) = poro(ix_left) * 0.1;

rock = struct('poro', poro);
src = addSource([], 3, -10);

%% Plot the watershed cells
newplot
plotGrid(G,'FaceColor',[0.95 0.95 0.95]); axis off;

%% Perform time of flight computation
tof = computeTimeOfFlight(state, G, rock, 'src', src, ...
   'maxTOF', 1', 'reverse', true);

clf,plotCellData(G,tof)