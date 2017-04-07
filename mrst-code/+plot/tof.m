function f = tof(CG, tof, showTofCentroids, saveName)
%PLOTTOF Summary of this function goes here
%   Detailed explanation goes here

scale = 255;
white = [255, 255, 255] ./ scale;
blueBrewer = [140, 160, 203] ./ scale;
greenBrewer = [102, 194, 165] ./ scale;
colors = [greenBrewer; white; blueBrewer];
x = [0, 128, 255];
map = interp1(x/255, colors, linspace(0,1,255));

%f = figure('position', [100, 100, 1000, 1000]);
f = figure();
colormap(map)
%axis off;
%h=colorbar;
%set(h, 'fontsize', 20);
%caxis([0, 2330]);
plotCellData(CG,tof, 'EdgeColor', 'none');

if showTofCentroids
    tofCent = text(CG.cells.centroids(:,1), CG.cells.centroids(:,2), ...
       num2str(ceil(tof)),'FontSize',18, 'HorizontalAlignment','center');
    set(tofCent,'BackgroundColor','w','EdgeColor','None');
end

end