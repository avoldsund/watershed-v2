function [CG, tof] = calculateTofRegularGrid(N)   
% Calculate time-of-flight using DEM

% From pre-processing: if a trap is spilling over into a cell at equal
% height, add that cell to the trap

%
scaleFluxes = true;
phi = 0.1;

% Load necessary data and compute geometry
landscapeName = strcat('landscapeRegularN', int2str(N), '.mat');
l = load(landscapeName);
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
[flux, ~] = util.setFlux(CG, nrOfTraps, trapNr, scaleFluxes);
state = struct('flux', flux);
rock = util.setPorosity(CG, nrOfTraps, phi);

% Calculate time-of-flight and subtract time it takes to fill src
maxTime = 10^8;
tof = computeTimeOfFlight(state, CG, rock, 'src', src, ...
   'maxTOF', maxTime, 'reverse', true);
tof = tof - min(tof);

end