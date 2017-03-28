%% Calculate hydrograph for a precipitation scenario in a DEM landscape

% Load necessary data and compute geometry
load('watershed.mat');
load('heights.mat');
load('traps.mat');
load('flowDirections.mat')
load('steepest.mat')

[nRows, nCols] = size(heights);
totalCells = nRows * nCols;
[heights, fd, ws, spillPairsIndices] = util.preProcessData(heights, flowDirections, watershed, spillPairs);
CG = util.createCoarseGrid(ws, heights, traps, nrOfTraps, spillPairs);
CG.cells.z = util.setHeightsCoarseGrid(CG, heights, trapHeights, nrOfTraps);

% Add flux field, state, rock and source
[src, trapNr] = util.getSource(CG, outlet, traps, nCols);
CG.cells.fd = util.getFlowDirections(CG, fd, nrOfTraps, spillPairsIndices);
[flux, faceFlowDirections] = util.setFlux(CG, nrOfTraps, trapNr);
state = struct('flux', flux);
rock = util.setPorosity(CG, nrOfTraps, 0.005);

% Calculate time-of-flight
maxTime = 500000;
tof = computeTimeOfFlight(state, CG, rock, 'src', src, ...
   'maxTOF', maxTime, 'reverse', true);

% Plot results
figure()
timeScale = 60;
tof = ceil(tof ./ timeScale);
clf,plotCellData(CG,tof, 'EdgeColor', 'none');
colormap(jet)
caxis([0, max(tof)])


%% Make disc hydrograph

% Direction and speed of disc precipitation
d = [1, 0];
v = 5;

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


%% Make moving front

% Define movement
d = [1, 0];
frontSize = 500;

minCoord = min(CG.faces.centroids);
minX = minCoord(1);
minY = minCoord(2);
maxCoord = max(CG.faces.centroids);
maxX = maxCoord(1);
maxY = maxCoord(2);

if d(1) ~= 0 % Move horizontally
    w = frontSize;
    l = maxY - minY;
    originX = minX - w;
    originY = minY;
    cornersX = [originX, originX, originX + w, originX + w];
    cornersY = [originY, originY + l, originY + l, originY];
    center = [originX + w/2, originY + l/2];
    if d(1) < 0 % Move west
        cornersX = cornersX + (maxX - minX) + w;
        center = [cornersX(1) + w/2, originY + l/2];
    end
    
else
    l = maxX - minX;
    w = frontSize;
    originX = minX;
    originY = minY - w;
    cornersX = [originX, originX, originX + l, originX + l];
    cornersY = [originY, originY + w, originY + w, originY];
    center = [originX + l/2, originY + w/2];
    
    if d(2) < 0 % Move south
        cornersY = cornersY + (maxY - minY) + w;
        center = [originX + l/2, cornersY(1) + w/2];
    end
end

A = 5;
v = 5;
gaussian = true;
front = struct('amplitude', A, 'velocity', v, 'direction', d, 'frontSize', frontSize, ...
    'center', center, 'cornersX', cornersX, 'cornersY', cornersY, 'gaussian', gaussian);

hydrograph = util.hydrographMovingFront(CG, tof, front);

figure();
plot(hydrograph);
