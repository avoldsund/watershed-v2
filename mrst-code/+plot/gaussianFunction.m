function f = gaussianFunction(A, R)
    %GAUSSIANFUNCTION Plots the Gaussian function used to construct the
    %hydrographs.
    %   GAUSSIANFUNCTION(AMPLITUDE, RADIUS) plots the Gaussian function
    %   that we use to construct our hydrographs.
    
    g = @(x, A, R) A * exp(-((x.^2)/(2 * (R/3)^2)));

    f = figure('position', [100, 100, 1000, 1000]);
    figure(f);
    x = linspace(-R, R, 100);
    plot(x, g(x, A, R), 'LineWidth', 5)
    set(gca, 'FontSize', 20)
    xlabel('Distance from center (m)')
    ylabel('I(x)')
    ylim([0, 10]);
    xlim([-30, 30]);
    %axis off

end

