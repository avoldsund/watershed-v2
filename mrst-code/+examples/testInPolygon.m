% Test the InPolygon function

nCols = 10;
nRows = 10;
stepSize = 10;
G = cartGrid([nCols, nRows], [nCols * stepSize, nRows * stepSize]);
G = computeGeometry(G);
cellCentroids = G.cells.centroids;
plotGrid(G)

% Circle moving diagonally over time
d = [1, 1];
v = 1;  % m/s

% Circle properties
c0 = [5, 5];
t = 0 : 0.01 : 2*pi;
r = 15;
c = c0;

T = 120;
% Iterate over time T
for i = 0:20:T
    x = cos(t) * r + c(1);
    y = sin(t) * r + c(2);
    
    in = inpolygon(cellCentroids(:, 1), cellCentroids(:, 2), x, y);

    hold on
    plot(x, y, 'r', 'Linewidth', 5)

    centroidsInside = cellCentroids(in, :);
    plot(centroidsInside(:, 1), centroidsInside(:, 2), 'b*');
    
    % Calculate where circle moves
    c(1) = c0(1) + d(1) * v * i;
    c(2) = c0(2) + d(2) * v * i;
end