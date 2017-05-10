%% Calculate hydrograph for a precipitation scenario in a DEM landscape

% If it doesn't work properly, change src in util.getSource

phi = 10^-7;
scaleFluxes = true;
[CG, tof] = calculateTof(phi, scaleFluxes);
tof = ceil(tof);
showTofCentroids = false;
cellIndices = false;
%f = plot.tof(CG, tof, showTofCentroids, cellIndices);


%% Remove largest value
maxTof = max(tof);
maxIndices = find(tof == maxTof);
tof(maxIndices) = 0;
newMax = max(tof);
tof(maxIndices) = newMax;

%% Uniform hydrograph

A = 10; % mm/hour
duration = 36000; % seconds
discharge = util.hydrographUniformFast(CG, tof, A, duration);

saveName = strcat('uH', 'P', int2str(A), 'D', int2str(duration));
saveName = strcat(saveName, '.eps');

h = plot.hydrograph(discharge, saveName);
%export_fig(saveName, h, '-eps')

%% Make moving front

%rIx = find(tof == max(tof));
%tof(rIx) = 0;

% Define movement
d = [0, -1]; % No need to normalize
frontSize = 1000;
offset = frontSize / 2;

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
    cornersY = [originY, originY + l, originY + l, originY];
    if d(1) > 0 % Move east
        cornersX = [originX, originX, originX + w, originX + w] + offset;
        center = [originX + offset + w/2, originY + l/2];
    else % Move west
        offset = (maxX - minX) + w - offset;
        cornersX = [originX, originX, originX + w, originX + w] + offset;
        center = [originX + offset + w/2, originY + l/2];
    end
    
else
    l = maxX - minX;
    w = frontSize;
    originX = minX;
    originY = minY - w;
    cornersX = [originX, originX, originX + l, originX + l];
    if d(2) > 0 % Move north
        cornersY = [originY, originY + w, originY + w, originY] + offset;
        center = [originX + l/2, originY + w/2 + offset];
    else % Move south
        offset = (maxY - minY) + w - offset;
        cornersY = [originY, originY + w, originY + w, originY] + offset;
        center = [originX + l/2, originY + w/2 + offset];
    end
end
corners = [cornersX; cornersY]';

intensity = 10; % mm/hour
v = 1; % m/s
gaussian = true;
maxTime = 1000000;

front = struct('amplitude', intensity,...
               'velocity', v,...
               'direction', d,...
               'frontSize', frontSize,...
               'center', center,...
               'corners', corners,...
               'gaussian', gaussian);

discharge = util.hydrographMovingFront(CG, tof, front, maxTime)%;, f);

saveName = 'frontSouth.eps';

h = plot.hydrograph(discharge, maxTime);
%export_fig(saveName, h, '-eps')


%% Make disc hydrograph
 
% Direction and speed of disc precipitation
d = [1, 1];
d = d ./ sqrt(sum(d.^2)) % Normalize
v = 0.1;
maxTime = 1300;

% Disc properties
c0 = [10, 10];
r = 30;
intensity = 1.11;
gaussian = true;
disc = struct('radius', r, ...
              'center', c0,...
              'direction', d, ...
              'velocity', v, ...
              'gaussian', gaussian, ...
              'amplitude', intensity);

discharge = util.hydrographMovingDisc(CG, tof, disc, maxTime, f);

saveName = strcat('discRadius30.eps');
h = plot.hydrograph(discharge, maxTime);

export_fig(saveName, h, '-eps')
