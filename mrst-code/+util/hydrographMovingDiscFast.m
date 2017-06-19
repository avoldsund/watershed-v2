function flow = hydrographMovingDiscFast(CG, tof, disc, timeStep, maxTime)
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
    [flow, in] = getDischarge(CG, tof, flow, disc, maxTime, cellArea, timeStep, t);
    t
    if t == 12
        hold on
        tVec = 0 : 0.01 : 2 * pi;
        x = cos(tVec) * disc.radius + disc.center(1);
        y = sin(tVec) * disc.radius + disc.center(2);
        plot(x, y, 'k', 'Linewidth', 3);
        cellCent = CG.parent.cells.centroids;
        centroidsInside = cellCent(in, :);
        plot(centroidsInside(:, 1), centroidsInside(:, 2), 'k*', 'MarkerSize', 24, 'LineWidth', 4);
    end


    disc = moveDisc(disc, timeStep);
    
    %if disc.center(1) > 33678 & disc.center(2) > 34678
    %    break 
    %end
    
    t = t + 1;
%     if t == 2
%         break
%     end
    
end

flow = flow * (10^-3) / 3600;

end

function disc = moveDisc(disc, timeStep)
    
    disc.center = bsxfun(@plus, disc.center, disc.direction * disc.velocity * timeStep);
    
end

function [flow, in] = getDischarge(CG, tof, flow, disc, maxTime, cellArea, timeStep, currentTime)
    
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
    validFrontIndices = tofInFront + currentTime * timeStep <= maxTime * timeStep;
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
    if currentTime == 1
       flow = zeros(maxTime, 1); 
    end
    
    cutOffNr = 25000;
    nrOfBoxes = ceil(size(tofInFront, 1)/cutOffNr);
    for i = 1:nrOfBoxes
        startT = (i-1) * cutOffNr + 1;
        if i == nrOfBoxes
            endT = size(tofInFront, 1);
        else
            endT = i * cutOffNr;
        end
        indices = reshape(repmat(tofInFront(startT:endT), 1, timeStep)', size(tofInFront(startT:endT), 1) * timeStep, 1);
        repetitions = repmat((((currentTime - 1)* timeStep+1):currentTime*timeStep), [1, size(tofInFront(startT:endT))])';
        ix = indices + repetitions;
        discharge = repelem((scale(startT:endT) .* cellArea)', timeStep);

        B = accumarray(ix, discharge);
        nzIndex = (B ~= 0);
        flow(nzIndex) = flow(nzIndex) + B(nzIndex);
    end
    
%     if currentTime == 1358
%        sum(v)* 1 / (3600*10^3)
%        in = false;
%     end

end