function f = watershed(landscapeName)
%WATERSHED Plots the watershed given the landscape object
%   F = WATERSHED(LANDSCAPENAME) takes a landscape object with name
%   LANDSCAPENAME, and plots the coarse grid CG. The traps are colored
%   blue, and the rest of the landscape is colored gray.
    
% Calculate coarse grid CG
l = load(landscapeName);
watershed = l.watershed;
faceLength = double(l.stepSize);
heights = l.heights;
traps = l.traps;
nrOfTraps = l.nrOfTraps;
flowDirections = l.flowDirections;
spillPairs = l.spillPairs;

[heights, ~, ws, ~] = util.preProcessData(heights, flowDirections, watershed, spillPairs);
CG = util.createCoarseGrid(ws, heights, traps, nrOfTraps, spillPairs, faceLength);

% Plot figure
f = figure('position', [100, 100, 1000, 1000]);
figure(f);

colors = zeros(2, 3);
colorLandscape = [0.86, 0.86, 0.86];
colorTraps = [0.08, 0.17, 0.55];
colors(1, :) = colorLandscape;
colors(2, :) = colorTraps;

colorIndices = zeros(CG.cells.num, 1);
colorIndices(CG.cells.num - nrOfTraps + 1:end) = 1;

plotGrid(CG, find(colorIndices == 0), 'faceColor', colors(1, :), 'EdgeColor', 'None');
plotGrid(CG, find(colorIndices), 'faceColor', colors(2, :), 'EdgeColor', 'None');
axis off
daspect([1 1 1])

end
