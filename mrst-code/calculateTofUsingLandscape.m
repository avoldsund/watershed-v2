%% Calculate time-of-flight using DEM

% From pre-processing: if a trap is spilling over into a cell at equal
% height, add that cell to the trap

% Load necessary data and compute geometry
load('watershed.mat');
load('heights.mat');
load('traps.mat');
load('flowDirections.mat')
load('steepest.mat')

[nRows, nCols] = size(heights);
totalCells = nRows * nCols;
%%
% Pre-process input data
heights = rot90(heights, -1);  % Fix 1d-indexing
fd = rot90(flowDirections, -1);  % Fix 1d-indexing
ws = util.mapCoordsToIndices(watershed', nCols, nRows);
spillPairsIndices = util.mapListOfCoordsToIndices(spillPairs, nCols, nRows);

% Create coarse grid and set heights
CG = util.createCoarseGrid(ws, heights, traps, nrOfTraps, spillPairs);
CG.cells.z = util.setHeightsCoarseGrid(CG, heights, trapHeights, nrOfTraps);

figure();
newplot
colorIndices = zeros(CG.cells.num, 1);
colorIndices(CG.cells.num - nrOfTraps + 1:end) = 1;

plotCellData(CG, CG.cells.z)
%plotGrid(CG,'FaceColor',[0.95 0.95 0.95]); axis off;
%plotCellData(CG,colorIndices,'EdgeColor','w','EdgeAlpha',.2);
%plotFaces(CG,(1:CG.faces.num)', 'FaceColor','none','LineWidth',2);
%colormap(.5*(colorcube(20) + ones(20,3))); axis off

%% Show cell/block indices
% In its basic form, the structure only represents topological information
% that specifies the relationship between blocks and block interfaces, etc.
% The structure also contains information of the underlying fine grid. Let
% us start by plotting cell/block indices
tg = text(CG.parent.cells.centroids(:,1), CG.parent.cells.centroids(:,2), ...
   num2str((1:CG.parent.cells.num)'),'FontSize',8, 'HorizontalAlignment','center');
tcg = text(CG.cells.centroids(:,1), CG.cells.centroids(:,2), ...
   num2str((1:CG.cells.num)'),'FontSize',16, 'HorizontalAlignment','center');
axis off;
set(tcg,'BackgroundColor','w','EdgeColor','none');
colormap(.5*jet+.5*ones(size(jet)));

%% Show face indices of fine/coarse grids
delete([tg; tcg]);
tg = text(CG.parent.faces.centroids(:,1), CG.parent.faces.centroids(:,2), ...
   num2str((1:CG.parent.faces.num)'),'FontSize',7, 'HorizontalAlignment','center');
tcg = text(CG.faces.centroids(:,1), CG.faces.centroids(:,2), ...
   num2str((1:CG.faces.num)'),'FontSize',12, 'HorizontalAlignment','center');
set(tcg,'BackgroundColor','w','EdgeColor','none');

%% Add flux field, state, rock and source
CG.cells.fd = util.getFlowDirections(CG, fd, nrOfTraps, spillPairsIndices);

flux = util.setFlux(CG, nrOfTraps);

state = struct('flux', flux);
rock = struct('poro', ones(CG.cells.num, 1));

% Find distance to CG.parent.cells.coords:
outlet = double(outlet);
%newOutlet = [10 * outlet(2), 60 - 10 * outlet(1)];
%distance = util.calculateEuclideanDist(CG.parent.cells.centroids, newOutlet);
%[M, I] = min(distance);
%src = CG.partition(I);
%src = src + 1;
src = addSource([], 10, -10);

% Perform time of flight computation
max_time = 500;
figure()

n = CG.cells.num - nrOfTraps + 1;
CG.cells.volumes(n:end) = CG.cells.volumes(n:end) * 0.01;
tof = computeTimeOfFlight(state, CG, rock, 'src', src, ...
   'maxTOF', max_time, 'reverse', true);


clf,plotCellData(CG,tof);

