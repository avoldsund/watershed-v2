function distance = calculateEuclideanDist(coords, coord)
%CALCULATEEUCLIDEANDIST Summary of this function goes here
%   Detailed explanation goes here
deltaX = bsxfun(@minus, coords(:, 1), coord(1));
deltaY = bsxfun(@minus, coords(:, 2), coord(2));

sum = deltaX.^2 + deltaY.^2;

distance = sqrt(sum);

end

