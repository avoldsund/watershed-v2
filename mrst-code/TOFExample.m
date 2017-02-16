%% Example for flow diagnostics

nCols = 3;
nRows = 3;
G = cartGrid([nCols, nRows], [30, 30]);
G = computeGeometry(G);
rock = makeRock(G, 0, 1);

field = 'flux';
value = zeros(9,1);

state = struct(field, value);
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

