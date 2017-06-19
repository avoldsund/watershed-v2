%% Calculate hydrograph for a precipitation scenario in a DEM landscape

% If it doesn't work properly, change src in util.getSource

phi = 0;
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
maxTime = 1E6;

front = struct('amplitude', intensity,...
               'velocity', v,...
               'direction', d,...
               'frontSize', frontSize,...
               'center', center,...
               'corners', corners,...
               'gaussian', gaussian);

timeStep = 1;

%hold on
%plot([front.corners(:, 1); front.corners(1, 1)], [front.corners(:, 2); front.corners(1, 2)], 'k-', 'Linewidth', 4);
%patch(front.corners(:, 1), front.corners(:, 2), color)

% front.center = [front.center(1) + front.velocity * front.direction(1, 1) * timeStep/2, ...
%     front.center(2) + front.velocity * front.direction(1, 2) * timeStep/2];
% front.corners = bsxfun(@plus, front.corners, front.velocity .* front.direction * timeStep/2);
discharge = util.hydrographMovingFrontFast(CG, tof, front, timeStep, maxTime);

%% Make disc hydrograph
 
% Direction and speed of disc precipitation
d = [1, 1];
d = d ./ sqrt(sum(d.^2)); % Normalize
v = 5;
maxTime = 1E6;

% Disc properties
%c0 = [29000, 18000];
%r = 1000;
%intensity = 27.35;
c0 = [10, 10];
r = 10;
intensity = 10;
gaussian = true;
disc = struct('radius', r, ...
              'center', c0,...
              'direction', d, ...
              'velocity', v, ...
              'gaussian', gaussian, ...
              'amplitude', intensity);

          
timeStep = 1;

% tVec = 0 : 0.01 : 2 * pi;
% x = cos(tVec) * disc.radius + disc.center(1);
% y = sin(tVec) * disc.radius + disc.center(2);
% plot(x, y, 'k', 'Linewidth', 3);
%position = [6000 7000 6000 6000];
%rectangle('Position', position, 'Curvature', [1 1], 'FaceColor', color)

%disc.center = [disc.center(1) + disc.velocity * disc.direction(1, 1) * timeStep/2, ...
%   disc.center(2) + disc.velocity * disc.direction(1, 2) * timeStep/2];
          
%dischargeSlow = util.hydrographMovingDisc(CG, tof, disc, maxTime);          
discharge = util.hydrographMovingDiscFast(CG, tof, disc, timeStep, maxTime);
