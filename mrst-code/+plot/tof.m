function f = tof(CG, tof, showTofCentroids, cellIndices)
%PLOTTOF Summary of this function goes here
%   Detailed explanation goes here

% Create custom color mapping
scale = 255;
greenDark = [0, 122, 0] ./ scale;
greenLight = [102, 194, 165] ./ scale;
white = [1, 1, 1];
grey = [0.5, 0.5, 0.5];
colors = [greenLight; grey; greenDark];

l1 = [239, 237, 245] ./ scale;
l2 = [188,189,220] ./ scale;
l3 = [117,107,177] ./ scale;
b1 = [222,235,247] ./ scale;
b2 = [158,202,225] ./ scale;
b3 = [49,130,189] ./ scale;

colors = [b1;b2;b3];

%colorsBW = [white; grey];
x = [0, 127, scale];
map = interp1(x/scale, colors, linspace(0, 1, scale));

% Create figure, set colormap, plot grid and face colors
f = figure('position', [100, 100, 1000, 1000]);

% Add frame/invisible line to front
color = [0.99, 0.99, 0.99];
%plot([10, 10, 50, 50, 10], [5, 55, 55, 5, 5], 'Color', color)

% Add frame/invisible line to disc
plot([0, 0, 60, 60, 0], [0, 60, 60, 0, 0], 'Color', color)

colormap(map)
%colormap(map);
plotCellData(CG, tof, 'EdgeColor', 'None');
plotGrid(CG, 'FaceColor', 'None', 'EdgeColor', 'None')

% Add colorbar, remove axis
%h = colorbar;
%set(h, 'fontsize', 30);
%caxis([0, 2330]);
axis off


if cellIndices
    textCells = text(CG.cells.centroids(:,1), CG.cells.centroids(:,2), ...
       num2str((1:CG.cells.num)'),'FontSize',24, 'HorizontalAlignment','center');
    set(textCells,'BackgroundColor','w','EdgeColor','none');
end

if showTofCentroids
    tofCent = text(CG.cells.centroids(:,1), CG.cells.centroids(:,2), ...
       num2str(ceil(tof)),'FontSize',24, 'HorizontalAlignment','center');
    set(tofCent,'BackgroundColor','w','EdgeColor','black');
end

end