function distance = calculateEuclideanDist(coords, coord)
%CALCULATEEUCLIDEANDIST Calculates Euclidean distance from a set of points
% to a coordinate
%   DISTANCE = CALCULATEEUCLIDEANDIST(COORDS, COORD) returns the Euclidean
%   distance from a set of coordinates COORDS to a given coordinate COORD.
%   COORDS is an Nx2-vector, while COORD is 1x2.

deltaX = bsxfun(@minus, coords(:, 1), coord(1));
deltaY = bsxfun(@minus, coords(:, 2), coord(2));

sum = deltaX.^2 + deltaY.^2;

distance = sqrt(sum);

end

