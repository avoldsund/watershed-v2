function [] = traps(CG, nrOfTraps)
    %TRAPS Summary of this function goes here
    %   Detailed explanation goes here
    
    colors = colormap(jet(double(nrOfTraps)));

    plotGrid(CG, 1:CG.cells.num-nrOfTraps, 'faceColor', 'w');
    for i = 1:nrOfTraps
        plotGrid(CG, CG.cells.num-nrOfTraps+i, 'faceColor', colors(i, :));
    end
end

