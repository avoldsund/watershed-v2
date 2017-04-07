%% Calculate hydrograph for a precipitation scenario in a DEM landscape

phi = 0.1;
scaleFluxes = true;
[CG, tof] = calculateTof(phi, scaleFluxes);
tof = ceil(tof);
showTofCentroids = false;
ftof = plot.tof(CG, tof, showTofCentroids, false);

%% Uniform hydrograph


amount = 10; % mm/hour
duration = 600;
discharge = util.hydrographUniform(CG, tof, amount, duration);

saveName = strcat('uniformHydrographNoScaling', 'P', int2str(amount), 'D', int2str(duration));
plot.hydrograph(discharge, saveName)

%% Make disc hydrograph
 
% Direction and speed of disc precipitation
d = [1, 0];
v = 3;

% Disc properties
c0 = [10, 10];
r = 10;
intensity = 10;
gaussian = true;
disc = struct('radius', r, 'center', c0,...
    'direction', d, 'speed', v, 'gaussian', gaussian, 'amplitude', intensity);

hydrograph = util.hydrographMovingDisc(CG, tof, disc);


%% Make moving front

% Define movement
d = [0, 1];
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
v = 1; % m/s
gaussian = false;
maxTime = 650;

front = struct('amplitude', intensity, 'velocity', v, 'direction', d, 'frontSize', frontSize, ...
    'center', center, 'corners', corners, 'gaussian', gaussian);

discharge = util.hydrographMovingFront(CG, tof, front, maxTime);

saveName = strcat('frontHydrograph', 'I', num2str(intensity), 'v', num2str(v), 'D', strcat(int2str(d(1)), int2str(d(2))));

h = plot.hydrograph(discharge, saveName);
export_fig(saveName, h, '-eps')

