function discharge = hydrographMovingFrontTimeStep(CG, tof, front, timeStep, maxTime)%, f)


cellArea = CG.faceLength * CG.faceLength;
flow = false;
in = true;
t = 1;

while any(in) == 1 & t < maxTime
    t
    [flow, in] = getDischarge(CG, tof, flow, front, maxTime, cellArea, timeStep, t);

    front = moveFront(front, timeStep);
    t = t + 1;
end

flow = flow * (10^-3) / 3600;

% Remove flow less than machine epsilon
discharge = zeros(maxTime, 1);
[~, jj, v] = find(flow);
validIx = v > eps;
discharge(jj(validIx)) = v(validIx);

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
    discharge = scale .* cellArea;
    newii = ones(1, n * timeStep);
    newjj = reshape(repmat(tofInFront, 1, timeStep), n * timeStep, 1)' + repmat((((currentTime - 1)* timeStep+1):currentTime*timeStep), 1, n);
    [ii, jj, v] = find(flow);
    ii = [ii, newii];
    jj = [jj, newjj];
    v = [v, repelem(discharge, timeStep)];
        
    flow = sparse(ii, jj, v);

end

function distance = getDistanceFromCenter(cellCentroids, front)
    
    % Distance is the distance from the center line for each cell centroid
    ix = find(front.direction ~= 0);
    distance = abs(cellCentroids(:, ix) - front.center(ix));
    tol = 1e-10;
    assert(any(distance - front.frontSize/2 > tol) == 0);
    
end