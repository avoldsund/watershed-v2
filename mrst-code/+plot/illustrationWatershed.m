% Plot an outline of the watershed and color the inside blue

set(0, 'DefaultFigureColor', [1 1 1])
f = figure('position', [100, 100, 1000, 1000]); hold on;
axis('off')
daspect([1 1 1])

color =  [0.549,0.6275,0.7961];
A = boundaryFaces(CG);
plotFaces(CG, A, 'LineWidth', 4)
%% Uniform precipitation
plotGrid(CG, 'FaceColor', color, 'EdgeColor', 'None')
