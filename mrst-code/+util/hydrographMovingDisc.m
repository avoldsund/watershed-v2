function discharge = hydrographMovingDisc(CG, tof, disc, maxTime)
%CALCULATEHYDROGRAPH Calculate the discharge at the outlet given a moving
%disc of precipitation.
%   DISCHARGE = HYDROGRAPHMOVINGFRONT(CG, TOF, DISC, MAXTIME) returns the
%   DISCHARGE from a watershed given a moving disc of precipitation, DISC.
%   The time-of-flight is given by the TOF variable, with a grid structure
%   CG. The last parameter MAXTIME describes the time interval
%   [0, MAXTIME] of the resulting hydrograph.

cellArea = CG.faceLength * CG.faceLength;
flow = false;
in = true;
t = 1;

while any(in) == 1 & t < maxTime
    [flow, in] = getDischarge(CG, tof, flow, disc, maxTime, cellArea, t);

    if t == 1358
        disp('hei')
        hold on
        tVec = 0 : 0.01 : 2 * pi;
        x = cos(tVec) * disc.radius + disc.center(1);
        y = sin(tVec) * disc.radius + disc.center(2);
        plot(x, y, 'b', 'Linewidth', 3);
        cellCent = CG.parent.cells.centroids;
        centroidsInside = cellCent(in, :);
        %plot(centroidsInside(:, 1), centroidsInside(:, 2), 'b*', 'MarkerSize', 16, 'LineWidth', 1);
        plot(centroidsInside(:, 1), centroidsInside(:, 2), 'b*', 'MarkerSize', 24, 'LineWidth', 4);
    end

    disc = moveDisc(disc);
    t = t + 1;
end

flow = flow * (10^-3) / 3600;

% Remove flow less than machine epsilon
discharge = zeros(maxTime, 1);
[~, jj, v] = find(flow);
validIx = v > eps;
discharge(jj(validIx)) = v(validIx);

end

function disc = moveDisc(disc)
    
    disc.center = bsxfun(@plus, disc.center, disc.direction * disc.velocity);
    
end

function [flow, in] = getDischarge(CG, tof, flow, disc, maxTime, cellArea, currentTime)
    
    % Cells in storm front
    t = 0 : 0.01 : 2 * pi;
    x = cos(t) * disc.radius + disc.center(1);
    y = sin(t) * disc.radius + disc.center(2);
    cellCent = CG.parent.cells.centroids;
    in = inpolygon(cellCent(:, 1), cellCent(:, 2), x, y);
    frontIndices = find(in);
    
    % Map the time-of-flight from coarse grid to fine grid
    tofInFront = tof(CG.partition(in));
    
    % Of these, only include cells where their precipitation reaches the outlet
    % before maxTime
    validFrontIndices = tofInFront + currentTime <= maxTime;
    frontIndices = frontIndices(validFrontIndices);
    tofInFront = tofInFront(validFrontIndices);
    
    n = size(tofInFront, 1);
    scale = ones(1, n);
    cellCentroids = cellCent(frontIndices, :);
    
    % Different cells get different intensities
    if disc.gaussian
        g = @(r, A, R) A * exp(-((r.^2)/(2 * (R/3)^2)));
        distanceFromCenter = util.calculateEuclideanDist(cellCentroids, disc.center);
        assert(any(distanceFromCenter > disc.radius) == 0);
        scale = g(distanceFromCenter, disc.amplitude, disc.radius)';
    else
        scale = scale * disc.amplitude;
    end
    
    % Add flow
    [ii, jj, v] = find(flow);
    ii = [ii, ones(1, n)];
    jj = [jj, tofInFront' + currentTime];
    v = [v, scale .* cellArea];
    
    if currentTime == 1358
       sum(v)* 1 / (3600*10^3)
       in = false;
    end
    if min(v) < 10^-4
        disp('hei')
    end
        
    flow = sparse(ii, jj, v);

end