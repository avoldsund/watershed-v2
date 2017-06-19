function f = hydrographDelta(discharge, timeStep)

set(0, 'DefaultFigureColor', [1 1 1])    
h = figure('position', [0, 0, 1000, 1000]);
figure(h);
hold on;
    
% Get average discharge for every Delta t-points, and which time they
% belong to
avDischarge = arrayfun(@(j) mean(discharge(j:j+timeStep-1)),1:timeStep:length(discharge)-timeStep+1)'; % the averaged vector
timeVec = (timeStep/2):timeStep:numel(discharge);
if size(timeVec, 2) > size(avDischarge, 1)
    timeVec = timeVec(1:end-1);
end

%timeVec = timeVec ./ 3600;
% Plot the approximated discharge
%plot(timeVec, avDischarge, '*-', 'LineWidth', 3, 'MarkerSize', 10)
plot(timeVec, avDischarge, 'LineWidth', 2)
xlabel('Time (hours)')
ylabel('Discharge{(}m^3/s)')
%ylim([0, 55])
%xlim([0, 195])
set(gca, 'FontSize', 34);
    
end
