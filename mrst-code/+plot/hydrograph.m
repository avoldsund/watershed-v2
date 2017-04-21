function h = hydrograph(hydrograph, maxTime)
    %HYDROGRAPH Summary of this function goes here
    %   Detailed explanation goes here
    
    color = [140, 160, 203] ./ 255;
    
    h = figure('position', [0, 0, 1000, 1000]);
    figure(h);
    
    plot(hydrograph, 'LineWidth', 3, 'Color', color)
    set(gca, 'FontSize', 30)
    xlabel('Time (s)')
    ylabel('Discharge{(}m^3/s)')
    %xlim([0, maxTime])
    ylim([0 1*10^-3])
    
end