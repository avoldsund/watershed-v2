function s = sensitivity(CG, tof)
    %SENSITIVITY Summary of this function goes here
    %   Detailed explanation goes here

    scale = 255;
    blueBrewer = [140, 160, 203] ./ scale;
    greenBrewer = [102, 194, 165] ./ scale;
    orangeBrewer = [252, 141, 98] ./ scale;
    
    h = histogram(tof, 10);
    s = figure('position', [0, 0, 1000, 1000]);
    figure(s);

    cmap = jet(h.NumBins);
    
    for i = 1:h.NumBins
        ix = tof >= h.BinEdges(i) & tof < h.BinEdges(i+1);
        plotGrid(CG, ix, 'faceColor', cmap(i, :));
    end
    axis('off')
    
    %colorIndices(ix) = 1;
    %colorIndices(ix2) = 2;
    %colors = zeros(3, 3);
    %colors(1, :) = greenBrewer;
    %colors(2, :) = blueBrewer;
    %colors(3, :) = orangeBrewer;
    %plotGrid(CG, find(colorIndices == 0), 'faceColor', colors(1, :));
    %plotGrid(CG, find(colorIndices == 2), 'faceColor', colors(2, :));
    %plotGrid(CG, find(colorIndices == 2), 'faceColor', colors(3, :));

    %plotCellData(CG, tof, 'EdgeColor', 'None');
    %plotGrid(CG, 'FaceColor', 'None', 'EdgeColor', 'None')
end
