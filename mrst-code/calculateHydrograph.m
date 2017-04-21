%% Calculate hydrograph for a precipitation scenario in a DEM landscape

phi = 0.1;
scaleFluxes = true;
[CG, tof] = calculateTof(phi, scaleFluxes);
tof = ceil(tof);
showTofCentroids = false;
cellIndices = false;
f = plot.tof(CG, tof, showTofCentroids, cellIndices);


%% Uniform hydrograph

A = 10; % mm/hour
duration = 3600; % seconds
discharge = util.hydrographUniform(CG, tof, A, duration);

saveName = strcat('uH', 'P', int2str(A), 'D', int2str(duration));
saveName = strcat(saveName, '.eps');

h = plot.hydrograph(discharge, saveName);
export_fig(saveName, h, '-eps')

%% Make moving front

% Define movement
d = [0, 1]; % No need to normalize
frontSize = 10;
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
v = 5; % m/s
gaussian = true;
maxTime = 650;

front = struct('amplitude', intensity,...
               'velocity', v,...
               'direction', d,...
               'frontSize', frontSize,...
               'center', center,...
               'corners', corners,...
               'gaussian', gaussian);

discharge = util.hydrographMovingFront(CG, tof, front, maxTime, f);

saveName = strcat('frontHydrograph', 'I', num2str(intensity), 'v', num2str(v), 'D', strcat(int2str(d(1)), int2str(d(2))));

h = plot.hydrograph(discharge, maxTime);
export_fig(saveName, h, '-eps')


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
