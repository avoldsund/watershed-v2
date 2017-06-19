function flow = hydrographMovingFrontFast(CG, tof, front, timeStep, maxTime)%, f)


cellArea = CG.faceLength * CG.faceLength;
flow = false;
in = true;
t = 1;

while any(in) == 1 & t < maxTime
    t
    [flow, in] = getDischarge(CG, tof, flow, front, maxTime, cellArea, timeStep, t);
    if t == 9
        hold on
        plot([front.corners(:, 1); front.corners(1, 1)], [front.corners(:, 2); front.corners(1, 2)], 'k-', 'Linewidth', 4);
        cellCent = CG.parent.cells.centroids;
        centroidsInside = cellCent(in, :);
        plot(centroidsInside(:, 1), centroidsInside(:, 2), 'k*', 'MarkerSize', 24, 'LineWidth', 4);
    end
    front = moveFront(front, timeStep);
    t = t + 1;
end

flow = flow * (10^-3) / 3600;

% Remove flow less than machine epsilon
%discharge = zeros(maxTime, 1);
%[~, jj, v] = find(flow);
%validIx = v > eps;
%discharge(jj(validIx)) = v(validIx);

end


function front = moveFront(front, timeStep)
    
    front.corners = bsxfun(@plus, front.corners, front.direction .* front.velocity * timeStep);
    front.center = bsxfun(@plus, front.center, front.direction * front.velocity * timeStep);

end

function [flow, in] = getDischarge(CG, tof, flow, front, maxTime, cellArea, timeStep, currentTime)
    
    % Cells in storm front
    cellCent = CG.parent.cells.centroids;
    in = inpolygon(cellCent(:, 1), cellCent(:, 2), front.corners(:, 1), front.corners(:, 2));
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
    if front.gaussian
        distance = getDistanceFromCenter(cellCentroids, front);
        g = @(r, A, R) A * exp(-((r.^2)/(2 * (R/3)^2)));
        frontRadius = front.frontSize/2;
        scale = g(distance, front.amplitude, frontRadius)';
    else
        scale = scale * front.amplitude;
    end
    
    % Add flow
    if currentTime == 1
       flow = zeros(maxTime, 1); 
    end
    
    cutOffNr = 100000;
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
    
end

function distance = getDistanceFromCenter(cellCentroids, front)
    
    % Distance is the distance from the center line for each cell centroid
    ix = find(front.direction ~= 0);
    distance = abs(cellCentroids(:, ix) - front.center(ix));
    tol = 1e-10;
    assert(any(distance - front.frontSize/2 > tol) == 0);
    
end