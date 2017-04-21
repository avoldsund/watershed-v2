function f = compareGaussian(amplitudes, radii)
    %COMPAREGAUSSIAN Compares a number of gaussian functions
    %   F = COMPAREGAUSSIAN(AMPLITUDES, RADII) plots Gaussian functions
    %   with different amplitudes and radii in the same figure, so they can
    %   be compared.
    
    g = @(x, A, R) A * exp(-((x.^2)/(2 * (R/3)^2)));

    f = figure('position', [100, 100, 1000, 1000]);
    figure(f);
    hold on
    
    for j = 1:size(amplitudes, 2)
        x = linspace(-radii(j), radii(j), 200);
        plot(x, g(x, amplitudes(j), radii(j)), 'LineWidth', 4,...
            'DisplayName', ['R = ', num2str(radii(j)), ' m'])
    end
    
    set(gca, 'FontSize', 20)
    xlabel('Distance from center (m)')
    ylabel('I(x) (mm/hour)')
    legend('show')
    
end