%% Plot example grid

%% Map from python-indices to Matlab-indices
%wsCoords =  [1, 1; 1, 2; 1, 3; 1, 4; 2, 2; 2, 3; 2, 4; 3, 2; 3, 3;];
nCols = 2;
nRows = 2;
%nCols = 6;
%nRows = 5;
h = 10;
lenCols = nCols * h;
lenRows = nRows * h;
%wsIndices = util.mapCoordsToIndices(wsCoords, nCols, nRows);

% Create 5x6-grid and remove some cells
G = cartGrid([nCols, nRows], [lenCols, lenRows]);
G = computeGeometry(G);
% Remove all cells in the grid not in the watershed
%rmCells = setdiff(1 : nCols * nRows, wsIndices);
%G = removeCells(G, rmCells);


newplot
plotGrid(G,'FaceColor',[0.95 0.95 0.95]); axis off;

%% Plot cell, face, and node numbers
% MRST uses a fully unstructured grid format in which cells, faces, and
% nodes, as well as their topological relationships, are represented
% explicitly. To illustrate, we extract the cell and face centroids as well as
% the coordinates of each node. These will be used for plotting the cells,
% faces and node indices, respectively.

c_cent = G.cells.centroids;
f_cent = G.faces.centroids;
coords = G.nodes.coords;

% Add circles around the centroids of each cell
hold on;
pltarg = {'MarkerSize',20,'LineWidth',2,'MarkerFaceColor',[.95 .95 .95]};

plot(c_cent(:,1), c_cent(:,2),'or',pltarg{:});

% Plot triangles around face centroids
plot(f_cent(:,1), f_cent(:,2),'sg',pltarg{:});

% Plot squares around nodes
plot(coords(:,1), coords(:,2),'db',pltarg{:});

legend({'Grid', 'Cell', 'Face', 'Node'}, 'Location', 'SouthOutside', 'Orientation', 'horizontal')

% Plot cell/face centroids and nodes
txtargs = {'FontSize',12,'HorizontalAlignment','left'};
text(c_cent(:,1)-0.04, c_cent(:,2), num2str((1:G.cells.num)'),txtargs{:});
text(f_cent(:,1)-0.045, f_cent(:,2), num2str((1:G.faces.num)'),txtargs{:});
text(coords(:,1)-0.075, coords(:,2), num2str((1:G.nodes.num)'),txtargs{:});

title('Grid structure')
hold off;