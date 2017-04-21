
%GETTESTGRID Summary of this function goes here
%   Detailed explanation goes here

% Create grid
N = 10; stepSize = 1;
G=cartGrid([N, N],[N * stepSize, N * stepSize]);
G=computeGeometry(G);

% Set flux, rock and source
srcStrength = 1;
src = addSource([], 1, -srcStrength);

[flux, ~] = util.setFlux(CG, nrOfTraps, trapNr, scaleFluxes);
state = struct('flux', flux);
rock = ones(N, 1);

% Calculate time-of-flight and subtract time it takes to fill src
maxTime = 10^8;
tof = computeTimeOfFlight(state, G, rock, 'src', src, ...
   'maxTOF', maxTime, 'reverse', true);
tof = tof - min(tof);