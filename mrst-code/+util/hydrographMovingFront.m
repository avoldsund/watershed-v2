function discharge = hydrographMovingFront(CG, tof, front, maxTime)
%CALCULATEHYDROGRAPH Calculate the discharge at the outlet given a moving
%front of precipitation.
%   FLOW = HYDROGRAPHMOVINGFRONT(CG, TOF, FRONT, MAXTIME) returns the FLOW from a
%   watershed given a moving front with precipitation, FRONT. The
%   time-of-flight is given by the TOF variable, with a grid structure CG.
%   The last parameter MAXTIME describes the time interval [0, MAXTIME] of
%   the resulting hydrograph.

cellArea = 10;
flow = false;
in = true;
time = 1;

while any(in) == 1 & time < maxTime

    [flow, in] = getDischarge(CG, tof, flow, front, maxTime, cellArea, time);

    %hold on
    %plot([front.corners(:, 1); front.corners(1, 1)], [front.corners(:, 2); front.corners(1, 2)], 'Linewidth', 3);
    %cellCent = CG.parent.cells.centroids;
    %centroidsInside = cellCent(in, :);
    %plot(centroidsInside(:, 1), centroidsInside(:, 2), 'b*', 'MarkerSize', 16);
    
    front = moveFront(front);
    time = time + 1;
end

flow = flow * (10^-3) / 3600;

% Remove flow less than machine epsilon
discharge = zeros(maxTime, 1);
[~, jj, v] = find(flow);
validIx = v > eps;
discharge(jj(validIx)) = v(validIx);

end


function front = moveFront(front)
    
    front.corners = bsxfun(@plus, front.corners, front.direction .* front.velocity);
    front.center = bsxfun(@plus, front.center, front.direction * front.velocity);

end

function [flow, in] = getDischarge(CG, tof, flow, front, maxTime, cellArea, currentTime)
    
    % Cells in storm front
    cellCent = CG.parent.cells.centroids;
    in = inpolygon(cellCent(:, 1), cellCent(:, 2), front.corners(:, 1), front.corners(:, 2));
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
    if front.gaussian
        g = @(r, A, R) A * exp(-((r.^2)/(2 * (R/3)^2)));
        ix = find(front.direction ~= 0);
        distanceFromCenterLine = abs(cellCentroids(:, ix) - front.center(ix));
        tol = 1e-10;
        assert(any(distanceFromCenterLine - front.frontSize/2 > tol) == 0);
        scale = g(distanceFromCenterLine, front.amplitude, front.frontSize/2)';
    else
        scale = scale * front.amplitude;
    end
    
    % Add flow
    [ii, jj, v] = find(flow);
    ii = [ii, ones(1, n)];
    jj = [jj, tofInFront' + currentTime];
    v = [v, scale .* cellArea];
    flow = sparse(ii, jj, v);

end