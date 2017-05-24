% Plot the problem with half a time step off
set(0, 'DefaultFigureColor', [1 1 1])

deltaT = {'60', '120', '240', '480', '960', '1920', '3840'};
velocity = '1';
%%
h = figure('position', [0, 0, 1000, 1000]);
figure(h);
hold on;
set(gca, 'FontSize', 24)

%% 
% Plot Delta t = 1 second
plot(d1.discharge, 'LineWidth', 2);
ylim([0, 180])
xlim([0, 7.5*10^5])

plot(d60.discharge, 'LineWidth', 2)
plot(d600.discharge, 'LineWidth', 2)
plot(d3600.discharge, 'LineWidth', 2)

xlabel('Time (s)')
ylabel('Discharge{(}m^3/s)')
legend('1', '60', '600', '3600');

%% Plot every n'th time
% Load reference solution
d1Name = strcat('dischargeNorth1sV', velocity, '.mat');
d1 = load(d1Name);
green = [0, 0.5, 0];
plot(d1.discharge, 'Color', green, 'LineWidth', 3);
maxTime = size(d1.discharge, 1);
lenDT = size(deltaT, 2);

% Initialize error norms
errors_l2 = zeros(1, lenDT);
errors_max = zeros(1, lenDT);
max_discharge = zeros(1, lenDT);
accumulated_discharge = zeros(lenDT+1, maxTime);

% Cutoff for artifact
artifact_cutoff = 65280; % Interval of 3820 will not include artifact [65280, 69120]
accumulated_discharge(1, artifact_cutoff:maxTime) = d1.discharge(artifact_cutoff:maxTime) * 1;

for i = 1:lenDT
    % Load discharge file
    timeStep = str2double(deltaT(i));
    fileName = strcat('dischargeNorth', deltaT(i), 'sV', velocity, '.mat');
    d = load(fileName{1});
    
    % Get average discharge for every Delta t-points, and which time they
    % belong to
    discharge = d.discharge;
    avDischarge = arrayfun(@(j) mean(discharge(j:j+timeStep-1)),1:timeStep:length(discharge)-timeStep+1)'; % the averaged vector
    timeVec = (timeStep/2):timeStep:numel(discharge);
    if size(timeVec, 2) > size(avDischarge, 1)
        timeVec = timeVec(1:end-1);
    end
    
    % Plot the approximated discharge
    %plot(timeVec, avDischarge, '*-', 'LineWidth', 3, 'MarkerSize', 10)
    plot(timeVec, avDischarge, 'LineWidth', 3)
    
    % Calculate error norms
    errors_l2(i) = sqrt(timeStep) * sqrt(sum(abs(d1.discharge(timeVec(timeVec > artifact_cutoff)) - avDischarge(timeVec > artifact_cutoff)).^2)) ...
        / norm(d1.discharge(artifact_cutoff:end)); %sqrt(sum(d1.discharge(artifact_cutoff:end).^2));
    errors_max(i) = max(abs(d1.discharge(timeVec(timeVec > artifact_cutoff)) - avDischarge(timeVec > artifact_cutoff)));
    max_discharge(i) = abs(max(d1.discharge(timeVec(timeVec > artifact_cutoff)) - max(avDischarge(timeVec > artifact_cutoff))));
    
    % Calculate accumulated discharge
    interpDischarge = interp1(timeVec, avDischarge, 1:maxTime);
    accumulated_discharge(i + 1, artifact_cutoff:maxTime) = interpDischarge(artifact_cutoff:maxTime) * 1;
end

xlabel('Time (s)')
ylabel('Discharge{(}m^3/s)')
ylim([0, 180])
xlim([0, 7.5*10^5])

legend(horzcat('1', deltaT), 'Location', 'NorthEast');

%% Accumulated flow
acc_disc = cumsum(accumulated_discharge, 2);
h = figure('position', [0, 0, 1000, 1000]); hold on;
green = [0, 0.5, 0];
plot(acc_disc(1, :), 'Color', green, 'LineWidth', 2)
for i = 2:lenDT+1
    plot(acc_disc(i, :), 'LineWidth', 2)
end
set(gca, 'FontSize', 24);
legend(horzcat('1', deltaT), 'Location', 'NorthWest')
xlim([0, 7.5*10^5])
xlabel('Time (s)')
ylab = ylabel('Accumulated flow{(}m^3)');

