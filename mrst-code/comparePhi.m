% Script for comparing time-of-flight for different phis

% Load data
l = load('landscapeTyrifjordenWatershed.mat');
watershed = l.watershed;
outlet = l.outlet;
faceLength = double(l.stepSize);
heights = l.heights;
traps = l.traps;
nrOfTraps = l.nrOfTraps;
trapHeights = l.trapHeights;
flowDirections = l.flowDirections;
spillPairs = l.spillPairs;
[~, nCols] = size(heights);

% Pre-process input data, create coarse grid and set heights
[heights, fd, ws, spillPairsIndices] = util.preProcessData(heights, flowDirections, watershed, spillPairs);
CG = util.createCoarseGrid(ws, heights, traps, nrOfTraps, spillPairs, faceLength);
CG.cells.z = util.setHeightsCoarseGrid(CG, heights, trapHeights, nrOfTraps);

% Set flux, rock and source
srcStrength = 1;
[src, trapNr] = util.getSource(CG, outlet, traps, nCols, srcStrength);
CG.cells.fd = util.getFlowDirections(CG, fd, nrOfTraps, spillPairsIndices);
[flux, ~] = util.setFlux(CG, nrOfTraps, trapNr, true);
state = struct('flux', flux);

%% 
phi = 10^-7;
rock = util.setPorosity(CG, nrOfTraps, phi);

maxTime = 10^8;
tof = computeTimeOfFlight(state, CG, rock, 'src', src, ...
   'maxTOF', maxTime, 'reverse', true, 'processCycles', true);
tof = tof - min(tof);

% Set cells with MAX_TIME to the second largest tof
maxTof = max(tof);
maxIndices = find(tof == maxTof);
tof(maxIndices) = 0;
newMax = max(tof);
tof(maxIndices) = newMax;

f = figure('position', [100, 100, 1000, 1000]);
figure(f);

plotCellData(CG, tof, 'EdgeColor', 'None');