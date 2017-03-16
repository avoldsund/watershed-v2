function flow = hydrographMovingFront(CG, tof, front)
%CALCULATEHYDROGRAPH Calculate the discharge at the outlet given a moving
%front of precipitation.
%   Detailed explanation goes here

maxTime = 3500;
cellArea = 10;
flow = NaN;

for time = 1:maxTime
    % Move precipitation disc and locate cells within
    [in, x, y, newCenter] = getCellsInMovingFront(CG, front, time);
    front.center = newCenter;
    if all(in == 0) && time > 1 % No cells in disc
       break 
    end
    
    %hold on
    %plot([x, x(1)], [y, y(1)], 'r', 'Linewidth', 3);
    %cellCent = CG.parent.cells.centroids;
    %centroidsInside = cellCent(in, :);
    %plot(centroidsInside(:, 1), centroidsInside(:, 2), 'b*');
    
    flow = updateFlow(CG, tof, flow, front, in, maxTime, cellArea, time);
end

flow = flow * (10^-3);%/3600;
end


function [in, cornersX, cornersY, newCenter] = getCellsInMovingFront(CG, front, currentTime)
    cellCent = CG.parent.cells.centroids;    
    cornersX = front.cornersX + front.direction(1) * front.velocity * (currentTime - 1);
    cornersY = front.cornersY + front.direction(2) * front.velocity * (currentTime - 1);
    
    if currentTime > 1
        newCenter = front.center + front.direction * front.velocity;
    else
        newCenter = front.center;
    end
    in = inpolygon(cellCent(:, 1), cellCent(:, 2), cornersX, cornersY);
end

function flow = updateFlow(CG, tof, flow, front, in, maxTime, cellArea, currentTime)
    % Update flow, of if flow does not exist yet, create the matrix
    
    tofOfIn = tof(CG.partition(in));
    contrFlow = tofOfIn(tofOfIn + currentTime <= maxTime);
    n = size(contrFlow, 1);
    scale = ones(1, n);
    
    indices = find(in);
    validIndices = indices(tofOfIn + currentTime <= maxTime);
    cellCentroids = CG.parent.cells.centroids(validIndices, :);
    
    if front.gaussian
        g = @(r, A, R) A * exp(-((r.^2)/(2 * (R/3)^2)));
        ix = find(front.direction ~= 0);
        distanceFromCenterLine = abs(cellCentroids(:, ix) - front.center(ix));
        assert(any(distanceFromCenterLine > front.frontSize/2) == 0);
        scale = g(distanceFromCenterLine, front.amplitude, front.frontSize/2)';
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