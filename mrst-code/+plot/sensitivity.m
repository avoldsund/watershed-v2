function [s, f, cbar] = sensitivity(CG, tof, numBins)
    %SENSITIVITY divides the time-of-flights in N time intervals and plots a
    %plots each interval in a separate color. A histogram is also created.
    %   [S, F, CBAR] = SENSITIVITY(CG, TOF, NUMBINS) creates a bar plot
    %   showing how the area is divided in terms of time-of-flight. The
    %   time intervals in the bar plot is used as basis for a
    %   time-of-flight plots where each bin has its own color.
    
    tofNew = tof / 3600;
    
    % Plot histogram using cell areas/bar plot
    s = figure('position', [0, 0, 1000, 1000]);
    figure(s);
    axH = axes('parent', s);
    hold(axH, 'on')
    [~, binEdges, binIndices] = histcounts(tofNew, 15);
    
    barAreas = zeros(numBins, 1);
    cmap = jet(numBins);
    colorIndices = binIndices;
    labels = {};
    for i = 1:numBins
        barAreas(i) = sum(CG.cells.volumes(binIndices == i));
        labels{i} =  strcat('[', num2str(binEdges(i)), ':', num2str(binEdges(i+1)), ')');
        bar(i, barAreas(i), 'parent', axH, 'facecolor', cmap(i, :));
    end
    
    xlim([0.6, numBins + 0.4])
    set(gca, 'XTickLabel', '', 'fontsize', 24);
    xlabel('Time intervals');
    ylabel('Total area (m^2)');
    
    % Plot time-of-flights using the histogram bins
    f = figure('position', [0, 0, 1000, 1000]);
    figure(f);
    plotCellData(CG, colorIndices, 'EdgeColor', 'None'); 
    colormap(cmap);
    axis('off')
    
    labels = binEdges;
    ytick = 1 : (numBins - 1) / numBins : numBins;
    cbar = colorbar('YTickLabel', labels, 'YTick', ytick, 'YLim', [1 numBins],...
        'fontsize', 20);
    ylabel(cbar, 'Time-of-flight in hours', 'fontSize', 20);
    daspect([1 1 1])
    
end
