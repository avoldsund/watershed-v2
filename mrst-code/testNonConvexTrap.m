nCols = 5;
nRows = 6;

% Create entire nRows x nCols-grid and remove cells outside watershed
stepSize = 10;
G = cartGrid([nCols, nRows], [nCols * stepSize, nRows * stepSize]);
G = computeGeometry(G);

% Combine traps and coarsen gridnrOfTraps
partition = [1,2,3,4,5,6,25,7,25,8,9,25,10,25,11,12,25,13,25,14,15,16,25,17,18, 19,20,21,22,23]';
partition = compressPartition(partition);
CG = generateCoarseGrid(G, partition);
CG = coarsenGeometry(CG);
nrOfTraps = 1;

figure();
newplot
colorIndices = zeros(CG.cells.num, 1);
colorIndices(CG.cells.num - nrOfTraps + 1:end) = 1:nrOfTraps;
plotCellData(CG, [1:24]','EdgeColor','w','EdgeAlpha',.2);

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
set(tcg,'BackgroundColor','r','EdgeColor','none');