set(0, 'DefaultFigureColor', [1 1 1])    
h = figure('position', [0, 0, 1000, 1000]);
figure(h);
hold on;


% Plot the approximated discharge
xlabel('Time (hours)')
ylabel('Discharge{(}m^3/s)')
set(gca, 'FontSize', 30);
time = 1:1E6;
time = time ./ 3600;
plot(time, twoHoursGaussian, 'LineWidth', 2)
hold on
plot(time, twoHoursUniform, 'LineWidth', 2)
ylim([0, 4])
xlim([0, 64])
legend('Gaussian', 'Uniform')