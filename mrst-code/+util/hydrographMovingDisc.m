function flow = hydrographMovingDisc(CG, tof, disc)
%CALCULATEHYDROGRAPH Calculate the discharge at the outlet given a moving
%disc of precipitation.
%   Detailed explanation goes here

maxTime = 2500;
cellArea = 10;
flow = NaN;
c = disc.center;

for time = 1:maxTime
    % Move precipitation disc and locate cells within
    [in, x, y, c] = getCellsInDisc(CG, disc, c, time);
    if all(in == 0) % No cells in disc
       break 
    end
    
    %hold on
    %plot(x, y, 'r', 'Linewidth', 3);
    %cellCent = CG.parent.cells.centroids;
    %centroidsInside = cellCent(in, :);
    %plot(centroidsInside(:, 1), centroidsInside(:, 2), 'b*');
    
    flow = updateFlow(CG, tof, flow, disc, in, c, maxTime, cellArea, time);
end

flow = flow * (10^-3)/60;
end


function [in, x, y, center] = getCellsInDisc(CG, disc, center, currentTime)
    % Moves the disc and finds the cells located within the moved disc
    
    cellCent = CG.parent.cells.centroids;    
    t = 0 : 0.01 : 2 * pi;
    center(1) = disc.center(1) + disc.direction(1) * disc.speed * (currentTime - 1);
    center(2) = disc.center(2) + disc.direction(2) * disc.speed * (currentTime - 1);
    x = cos(t) * disc.radius + center(1);
    y = sin(t) * disc.radius + center(2);
    in = inpolygon(cellCent(:, 1), cellCent(:, 2), x, y);
end

function flow = updateFlow(CG, tof, flow, disc, in, c, maxTime, cellArea, currentTime)
    % Update flow, of if flow does not exist yet, create the matrix
    
    tofOfIn = tof(CG.partition(in));
    contrFlow = tofOfIn(tofOfIn + currentTime <= maxTime);
    n = size(contrFlow, 1);
    scale = ones(1, n);
    
    indices = find(in);
    validIndices = indices(tofOfIn + currentTime <= maxTime);
    cellCentroids = CG.parent.cells.centroids(validIndices, :);
    
    if disc.gaussian
        g = @(r, A, R) A * exp(-((r.^2)/(2 * (R/3)^2)));
        distanceFromCenter = util.calculateEuclideanDist(cellCentroids, c);
        assert(any(distanceFromCenter > disc.radius) == 0);
        scale = g(distanceFromCenter, disc.amplitude, disc.radius)';
    end
    
    if ~isnan(flow)
        [ii, jj, v] = find(flow);
        ii = [ii, ones(1, n)];
        jj = [jj, contrFlow' + currentTime];
        v = [v, scale .* cellArea];
    else
        ii = ones(1, n);
        jj = contrFlow' + currentTime;
        v = scale .* cellArea;
    end
    flow = sparse(ii, jj, v);
end