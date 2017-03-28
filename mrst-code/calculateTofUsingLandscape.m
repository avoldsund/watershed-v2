%% Calculate time-of-flight using DEM

% From pre-processing: if a trap is spilling over into a cell at equal
% height, add that cell to the trap

% Colors
scale = 255;
blueBrewer = [140, 160, 203] ./ scale;
greenBrewer = [102, 194, 165] ./ scale;
orangeBrewer = [252, 141, 98] ./scale;

% Load necessary data and compute geometry
load('watershed.mat');
load('heights.mat');
load('traps.mat');
load('flowDirections.mat')
load('steepest.mat')

[nRows, nCols] = size(heights);
totalCells = nRows * nCols;

% Pre-process input data, create coarse grid and set heights
sideLength = 10;
[heights, fd, ws, spillPairsIndices] = util.preProcessData(heights, flowDirections, watershed, spillPairs);
CG = util.createCoarseGrid(ws, heights, traps, nrOfTraps, spillPairs, sideLength);
CG.cells.z = util.setHeightsCoarseGrid(CG, heights, trapHeights, nrOfTraps);

% Plot landscape with traps
figure();
newplot
colorIndices = zeros(CG.cells.num, 1);
%timecolorIndices(CG.cells.num - nrOfTraps + 1:end) = 1:nrOfTraps;
%plotCellData(CG,colorIndices,'EdgeColor',greenBrewer,'EdgeAlpha',3);
plotGrid(CG)
%% Show cell/block indices
% In its basic form, the structure only represents topological information
% that specifies the relationship between blocks and block interfaces, etc.
% The structure also contains information of the underlying fine gr id. Let
% us start by plotting cell/block indices

%tg = text(CG.parent.cells.centroids(:,1), CG.parent.cells.centroids(:,2), ...
%   num2str((1:CG.parent.cells.num)'),'FontSize',8, 'HorizontalAlignment','center');
tcg = text(CG.cells.centroids(:,1), CG.cells.centroids(:,2), ...
   num2str((1:CG.cells.num)'),'FontSize',16, 'HorizontalAlignment','center');
set(tcg,'BackgroundColor','w','EdgeColor','None');
colormap(.5*jet+.5*ones(size(jet)));

%% Show face indices of fine/coarse grids
%delete([tg; tcg]);
%tg = text(CG.parent.faces.centroids(:,1), CG.parent.faces.centroids(:,2), ...
%   num2str((1:CG.parent.faces.num)'),'FontSize',7, 'HorizontalAlignment','center');
tcg = text(CG.faces.centroids(:,1), CG.faces.centroids(:,2), ...
   num2str((1:CG.faces.num)'),'FontSize',12, 'HorizontalAlignment','center');
set(tcg,'BackgroundColor','w','EdgeColor','none');

%% Perform time-of-flight

% Add flux field, state, rock and source
srcStrength = 1;
[src, trapNr] = util.getSource(CG, outlet, traps, nCols, srcStrength);
CG.cells.fd = util.getFlowDirections(CG, fd, nrOfTraps, spillPairsIndices);
[flux, faceFlowDirections] = util.setFlux(CG, nrOfTraps, trapNr);
state = struct('flux', flux);
rock = util.setPorosity(CG, nrOfTraps, 1);

% Calculate time-of-flight
maxTime = 3000;
tof = computeTimeOfFlight(state, CG, rock, 'src', src, ...
   'maxTOF', maxTime, 'reverse', true);

% Plot results
figure()
%timeScale = 60;
%tof = ceil(tof ./ timeScale);
clf,plotCellData(CG,tof, 'EdgeColor', 'none');
colormap(jet)
%caxis([0, 120000/timeScale])

%% Make uniform hydrograph
timeScale = 60;
amount = 1;
duration = 5;
tof = ceil(tof ./ timeScale);

hydrograph = util.hydrographUniform(CG, tof, amount, duration);
plot(hydrograph)

if timeScale == 60
    xlabel('Time (minutes)')
else
    xlabel('Time (hours)')
end
ylabel('Flow m^3/s')