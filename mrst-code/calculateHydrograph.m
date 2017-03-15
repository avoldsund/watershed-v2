%% Calculate time-of-flight using DEM

% From pre-processing: if a trap is spilling over into a cell at equal
% height, add that cell to the trap

% Load necessary data and compute geometry
load('watershed.mat');
load('heights.mat');
load('traps.mat');
load('flowDirections.mat')
load('steepest.mat')

[nRows, nCols] = size(heights);
totalCells = nRows * nCols;
[heights, fd, ws, spillPairsIndices] = util.preProcessData(heights, flowDirections, watershed, spillPairs);

% Create coarse grid and set heights
CG = util.createCoarseGrid(ws, heights, traps, nrOfTraps, spillPairs);
CG.cells.z = util.setHeightsCoarseGrid(CG, heights, trapHeights, nrOfTraps);

% Add flux field, state, rock and source
CG.cells.fd = util.getFlowDirections(CG, fd, nrOfTraps, spillPairsIndices);
flux = util.setFlux(CG, nrOfTraps);

% Add flux and porosity. Make porosity smaller for lakes, so that they're
% already full
state = struct('flux', flux);
rock = struct('poro', ones(CG.cells.num, 1));
n = CG.cells.num - nrOfTraps + 1;
oneVec = ones(nrOfTraps, 1);
rock.poro(n:end) = oneVec * 0.01;

% Find source:
outlet = double(outlet);
newOutlet = [10 * outlet(2), 10 * nCols - 10 * outlet(1)];
distance = util.calculateEuclideanDist(CG.parent.cells.centroids, newOutlet);
[M, I] = min(distance);
src = CG.partition(I);
src = src + 1;
%src = 10;
src = addSource([], src, -10);

% Perform time of flight computation
max_time = 500000;
tof = computeTimeOfFlight(state, CG, rock, 'src', src, ...
   'maxTOF', max_time, 'reverse', true);

% Plot results
figure()
timeScale = 60;
tof = ceil(tof ./ timeScale);
clf,plotCellData(CG,tof, 'EdgeColor', 'none');
colormap(jet)
caxis([0, 120000/timeScale])

%% Make disc hydrograph
%timeScale = 60;
%tof = ceil(tof ./ timeScale);

% Direction and speed of disc precipitation
d = [1, 1.3];
v = 10;

% Disc properties
c0 = [10, 10];
r = 500;
A = 10;
gaussian = true;
disc = struct('radius', r, 'center', c0,...
    'direction', d, 'speed', v, 'gaussian', gaussian, 'amplitude', A);

hydrograph = util.hydrographMovingDisc(CG, tof, disc);
figure();
plot(hydrograph);