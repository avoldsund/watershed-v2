% Plot the problem with half a time step off
set(0, 'DefaultFigureColor', [1 1 1])    
h = figure('position', [0, 0, 1000, 1000]);
figure(h);
hold on;
set(gca, 'FontSize', 24)

d1 = load('dischargeNorth1sV1.mat');
d600 = load('dischargeNorth600sV1Old.mat');
d3600 = load('dischargeNorth3600sV1Old.mat');

% Plot Delta t = 1 second
plot(d1.discharge, 'LineWidth', 1.5);
ylim([0, 180])
xlim([0, 7.5*10^5])

% Plot Delta t = 600 seconds
discharge = d600.discharge; timeStep = 600;
b = arrayfun(@(i) mean(discharge(i:i+timeStep-1)),1:timeStep:length(discharge)-timeStep+1)'; % the averaged vector
c = (timeStep/2):timeStep:numel(discharge);
c = c(1:end-1);
plot(c, b, 'LineWidth', 1.5)

% Plot Delta t = 3600 seconds
discharge = d3600.discharge; timeStep = 3600;
b = arrayfun(@(i) mean(discharge(i:i+timeStep-1)),1:timeStep:length(discharge)-timeStep+1)'; % the averaged vector
c = (timeStep/2):timeStep:numel(discharge);
c = c(1:end-1);
plot(c, b, 'LineWidth', 1.5)

xlabel('Time (s)')
ylabel('Discharge{(}m^3/s)')
legend('1', '600', '3600');