function h = hydrograph(hydrograph, maxTime)
    %HYDROGRAPH Summary of this function goes here
    %   Detailed explanation goes here
    
    color = [140, 160, 203] ./ 255;
    
    h = figure('position', [0, 0, 1000, 1000]);
    figure(h);
    
    plot(hydrograph, 'LineWidth', 3, 'Color', color)
    set(gca, 'FontSize', 28)
    set(gca,'XTickLabel', 20:20:220, 'XTicks', 0:72000:800000)
    xlabel('Time (s)')
    ylabel('Discharge{(}m^3/s)')
    %ylabel('Discharge{(}m^3/s)')
    %xlim([0, maxTime])
    %ylim([0 9*10^-4])
    
end