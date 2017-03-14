function flow = hydrographMovingDisc(CG, tof, disc)
%CALCULATEHYDROGRAPH Summary of this function goes here
%   Detailed explanation goes here

% Iterate over time
maxTime = 2000;

cellCent = CG.parent.cells.centroids;
t = 0 : 0.01 : 2*pi;
cellArea = 10;

% First timestep
time = 1;
x = cos(t) * disc.radius + disc.center(1);
y = sin(t) * disc.radius + disc.center(2);
in = inpolygon(cellCent(:, 1), cellCent(:, 2), x, y);
hold on
plot(x, y, 'r', 'Linewidth', 5);
%centroidsInside = cellCent(in, :);
%plot(centroidsInside(:, 1), centroidsInside(:, 2), 'b*');

% Cells contributes flow if tof is less than maxTime
tofOfIn = tof(CG.partition(in));
contrFlow = tofOfIn(tofOfIn < maxTime);
n = size(contrFlow, 1);

ii = ones(1, n);
jj = contrFlow' + time;
v = cellArea * ones(1, n);
flow = sparse(ii, jj, v);

c = NaN;
for time = 2:40
    % Move precipitation disc and locate cells within
    c(1) = disc.center(1) + disc.direction(1) * disc.speed * (time - 1);
    c(2) = disc.center(2) + disc.direction(2) * disc.speed * (time - 1);
    x = cos(t) * disc.radius + c(1);
    y = sin(t) * disc.radius + c(2);
    in = inpolygon(cellCent(:, 1), cellCent(:, 2), x, y);
    
    hold on
    plot(x, y, 'r', 'Linewidth', 3);
    %centroidsInside = cellCent(in, :);
    %plot(centroidsInside(:, 1), centroidsInside(:, 2), 'b*');
    
    % Get tof from cells in coarse grid
    tofOfIn = tof(CG.partition(in)); 
    contrFlow = tofOfIn(tofOfIn < maxTime);
    
    n = size(contrFlow, 1);

    ii = [ii, ones(1, n)];
    jj = [jj, contrFlow' + time];
    v = [v, cellArea * ones(1, n)];
    flow = sparse(ii, jj, v);
end

flow = flow * (10^-3)/3600;

end
