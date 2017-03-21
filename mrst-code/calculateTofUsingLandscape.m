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

% Pre-process input data, create coarse grid and set heights
[heights, fd, ws, spillPairsIndices] = util.preProcessData(heights, flowDirections, watershed, spillPairs);
CG = util.createCoarseGrid(ws, heights, traps, nrOfTraps, spillPairs);
CG.cells.z = util.setHeightsCoarseGrid(CG, heights, trapHeights, nrOfTraps);

% Plot landscape with traps
figure();
newplot
colorIndices = zeros(CG.cells.num, 1);
colorIndices(CG.cells.num - nrOfTraps + 1:end) = 1:nrOfTraps;
plotCellData(CG,colorIndices,'EdgeColor','w','EdgeAlpha',.2);

%% Show cell/block indices
% In its basic form, the structure only represents topological information
% that specifies the relationship between blocks and block interfaces, etc.
% The structure also contains information of the underlying fine gr id. Let
% us start by plotting cell/block indices
tg = text(CG.parent.cells.centroids(:,1), CG.parent.cells.centroids(:,2), ...
   num2str((1:CG.parent.cells.num)'),'FontSize',8, 'HorizontalAlignment','center');
tcg = text(CG.cells.centroids(:,1), CG.cells.centroids(:,2), ...
   num2str((1:CG.cells.num)'),'FontSize',16, 'HorizontalAlignment','center');
set(tcg,'BackgroundColor','w','EdgeColor','none');
colormap(.5*jet+.5*ones(size(jet)));

%% Show face indices of fine/coarse grids
delete([tg; tcg]);
tg = text(CG.parent.faces.centroids(:,1), CG.parent.faces.centroids(:,2), ...
   num2str((1:CG.parent.faces.num)'),'FontSize',7, 'HorizontalAlignment','center');
tcg = text(CG.faces.centroids(:,1), CG.faces.centroids(:,2), ...
   num2str((1:CG.faces.num)'),'FontSize',12, 'HorizontalAlignment','center');
set(tcg,'BackgroundColor','w','EdgeColor','none');

%% Perform time-of-flight

% Add flux field, state, rock and source
[src, trapNr] = util.getSource(CG, outlet, traps, nCols);
CG.cells.fd = util.getFlowDirections(CG, fd, nrOfTraps, spillPairsIndices);
[flux, faceFlowDirections] = util.setFlux(CG, nrOfTraps, trapNr);
state = struct('flux', flux);
rock = util.setPorosity(CG, nrOfTraps, 0.005);

% Calculate time-of-flight
max_time = 200000;
tof = computeTimeOfFlight(state, CG, rock, 'src', src, ...
   'maxTOF', max_time, 'reverse', true);

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
tof = floor(tof ./ timeScale);

hydrograph = util.hydrographUniform(CG, tof, amount, duration);
plot(hydrograph)

if timeScale == 60
    xlabel('Time (minutes)')
else
    xlabel('Time (hours)')
end
ylabel('Flow m^3/s')


%% Make disc hydrograph
%timeScale = 60;
%tof = ceil(tof ./ timeScale);

% Direction and speed of disc precipitation
d = [1, 1];
v = 100;

% Disc properties
c0 = [10, 10];
r = 500;
amount = 1;
disc = struct('radius', r, 'center', c0, 'amount', amount,...
    'direction', d, 'speed', v);

hydrograph = util.hydrographMovingDisc(CG, tof, disc);


%% 