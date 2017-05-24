function h = hydrograph(hydrograph, maxTime)
    %HYDROGRAPH Summary of this function goes here
    %   Detailed explanation goes here
    
    color = [140, 160, 203] ./ 255;
    
    h = figure('position', [0, 0, 1000, 1000]);
    figure(h);
    
    % Add zeros at the end of hydrograph
    padWithZeros = zeros(maxTime, 1);
    padWithZeros(1:size(hydrograph, 1)) = hydrograph;
    hydrograph = padWithZeros;
    
    plot(hydrograph, 'LineWidth', 3, 'Color', color)
    set(gca, 'FontSize', 24)
    %ticks = 0:72000:745000;
    %set(gca,'XTickLabel', 0:20:size(ticks, 2)*20, 'XTick', ticks)
    xlabel('Time (s)')
    ylabel('Discharge{(}m^3/s)')
    %ylabel('Discharge{(}m^3/s)')
    xlim([0, maxTime])
    ylim([0 1000])
    
end