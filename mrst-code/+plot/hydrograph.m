function h = hydrograph(hydrograph, saveName)
    %HYDROGRAPH Summary of this function goes here
    %   Detailed explanation goes here
    
    color = [140, 160, 203] ./ 255;
    
    h = figure('position', [100, 100, 1000, 1000]);
    figure(h);
    
    plot(hydrograph, 'LineWidth', 3, 'Color', color)
    set(gca, 'FontSize', 26)
    xlabel('Time (s)')
    ylabel('Discharge{(}m^3/s)')
    %ylim([0 9*10^-5])
    %xlim([0, 1200])
end

