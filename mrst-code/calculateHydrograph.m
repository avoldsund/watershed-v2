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
%max_time = 500;
max_time = 150000;
tof = computeTimeOfFlight(state, CG, rock, 'src', src, ...
   'maxTOF', max_time, 'reverse', true);

% Plot results
figure()
timeScale = 3600;
tof = ceil(tof ./ timeScale);
clf,plotCellData(CG,tof, 'EdgeColor', 'none');
colormap(jet)
caxis([0, 120000/timeScale])
%caxis([0, 120000]);

%% Make disc hydrograph
%timeScale = 60;
%tof = ceil(tof ./ timeScale);

% Direction and speed of disc precipitation
d = [1, 1.3];
v = 80;

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
v = 5*60;
gaussian = true;
front = struct('amplitude', A, 'velocity', v, 'direction', d, 'frontSize', frontSize, ...
    'center', center, 'cornersX', cornersX, 'cornersY', cornersY, 'gaussian', gaussian);

hydrograph = util.hydrographMovingFront(CG, tof, front);

figure();
plot(hydrograph);