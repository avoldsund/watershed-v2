%% Plot grids

% Load necessary data and compute geometry
load('watershed.mat');
load('heights.mat');
load('traps.mat');
load('flowDirections.mat')
load('steepest.mat')

[nRows, nCols] = size(heights);
totalCells = nRows * nCols;

% Colors
scale = 255;
blueBrewer = [140, 160, 203] ./ scale;
greenBrewer = [102, 194, 165] ./ scale;
orangeBrewer = [252, 141, 98] ./scale;
blackNew = [eps, 0, 0];
whiteNew = [1 - eps, 1, 1];

% Pre-process input data, create coarse grid and set heights
[heights, fd, ws, spillPairsIndices] = util.preProcessData(heights, flowDirections, watershed, spillPairs);
stepSize = 10;
CG = util.createCoarseGrid(ws, heights, traps, nrOfTraps, spillPairs, stepSize);
CG.cells.z = util.setHeightsCoarseGrid(CG, heights, trapHeights, nrOfTraps);

%% Plot landscape with traps
figure();
newplot
colorIndices = zeros(CG.cells.num, 1);
%timecolorIndices(CG.cells.num - nrOfTraps + 1:end) = 1:nrOfTraps;
plotCellData(CG,colorIndices,'EdgeColor',greenBrewer,'EdgeAlpha',0.5);

%% Show cell/block indices and face indices of fine grid
% In its basic form, the structure only represents topological information
% that specifies the relationship between blocks and block interfaces, etc.
% The structure also contains information of the underlying fine gr id. Let
% us start by plotting cell/block indices
f = figure('position', [100, 100, 1000, 1000]);
figure(f);
plotGrid(CG.parent, 'FaceColor', greenBrewer);
axis off

textCells = text(CG.parent.cells.centroids(:,1), CG.parent.cells.centroids(:,2), ...
   num2str((1:CG.parent.cells.num)'),'FontSize',24, 'HorizontalAlignment','center');
set(textCells,'BackgroundColor','w','EdgeColor','None');

textFaces = text(CG.parent.faces.centroids(:,1), CG.parent.faces.centroids(:,2), ...
   num2str((1:CG.parent.faces.num)'),'FontSize',16, 'HorizontalAlignment','center', 'Color', whiteNew);
set(textFaces,'BackgroundColor',blackNew,'EdgeColor','none');

print(f, '-depsc', 'gridStructure.eps')


%% Show cell/block indices and show face indices of coarse grid
% In its basic form, the structure only represents topological information
% that specifies the relationship between blocks and block interfaces, etc.
% The structure also contains information of the underlying fine gr id. Let
% us start by plotting cell/block indices

f = figure('position', [100, 100, 1000, 1000]);
figure(f);

colorIndices = zeros(CG.cells.num, 1);
colorIndices(CG.cells.num - nrOfTraps + 1:end) = 1;
colors = zeros(2, 3);
colors(1, :) = greenBrewer;
colors(2, :) = blueBrewer;
plotGrid(CG, find(colorIndices == 0), 'faceColor', colors(1, :));
plotGrid(CG, find(colorIndices), 'faceColor', colors(2, :));

axis off

textCells = text(CG.cells.centroids(:,1), CG.cells.centroids(:,2), ...
   num2str((1:CG.cells.num)'),'FontSize',24, 'HorizontalAlignment','center');
set(textCells,'BackgroundColor','w','EdgeColor','none');

%textFaces = text(CG.faces.centroids(:,1), CG.faces.centroids(:,2), ...
%   num2str((1:CG.faces.num)'),'FontSize',16, 'HorizontalAlignment','center', 'Color', whiteNew);
%set(textFaces,'BackgroundColor',blackNew,'EdgeColor','none');


print(f, '-dsvg', 'coarseGridStructure.svg')


%% Show time-of-flight

% Add flux field, state, rock and source
srcStrength = 1;
[src, trapNr] = util.getSource(CG, outlet, traps, nCols, srcStrength);
CG.cells.fd = util.getFlowDirections(CG, fd, nrOfTraps, spillPairsIndices);
[flux, faceFlowDirections] = util.setFlux(CG, nrOfTraps, trapNr);
state = struct('flux', flux);
rock = util.setPorosity(CG, nrOfTraps, 0.01);

% Calculate time-of-flight and subtract time it takes to fill src
maxTime = 3000;
tof = computeTimeOfFlight(state, CG, rock, 'src', src, ...
   'maxTOF', maxTime, 'reverse', true);
tof = tof - min(tof);

% Plot results
f = figure('position', [100, 100, 1000, 1000]);
figure(f);
%timeScale = 60;
%tof = ceil(tof ./ timeScale);
clf,plotCellData(CG,tof, 'EdgeColor', 'none');
colormap(jet)
tofCent = text(CG.cells.centroids(:,1), CG.cells.centroids(:,2), ...
   num2str(ceil(tof)),'FontSize',16, 'HorizontalAlignment','center');
set(tofCent,'BackgroundColor','w','EdgeColor','None');
%colormap(.5*jet+.5*ones(size(jet)));
colorbar();
axis off;

%caxis([0, 2330])

print(f, '-depsc', 'tofPhiOneHundreth.eps')