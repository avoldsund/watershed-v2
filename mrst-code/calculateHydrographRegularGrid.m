%% Calculate hydrograph for a precipitation scenario in a DEM landscape

N = 100;
[CG, tof] = calculateTofRegularGrid(N);
tof = ceil(tof);
showTofCentroids = false;
cellIndices = false;
f = plot.tof(CG, tof, showTofCentroids, cellIndices);

%% Make disc hydrograph
 
% Direction and speed of disc precipitation
d = [1, -1]; % Directional vector
d = d ./ sqrt(sum(d.^2)); % Normalize
v = 1;
maxTime = 2000;

% Disc properties
c0 = [22, 982]; % 500x500
%c0 = [24, 984]; % 250x250
%c0 = [30, 990]; % 100x100
%c0 = [40, 1000]; % 50x50

r = 20;
intensity = 2.5;
gaussian = true;
disc = struct('radius', r, ...
              'center', c0,...
              'direction', d, ...
              'velocity', v, ...
              'gaussian', gaussian, ...
              'amplitude', intensity);

discharge = util.hydrographMovingDisc(CG, tof, disc, maxTime);

saveName = strcat('diskN500');
h = plot.hydrograph(discharge, maxTime);

export_fig(saveName, h, '-eps')
