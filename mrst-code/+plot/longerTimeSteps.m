function [ output_args ] = longerTimeSteps()
    %LONGERTIMESTEPS Summary of this function goes here
    %   Detailed explanation goes here
    
d1 = load('dischargeNorth1sV1.mat');

hold on
normErrors = [0, 0, 0, 0];
errorsMax = [0, 0, 0, 0];
maxAtTime = [0, 0, 0, 0];
times = [60, 600, 1800, 3600];
for i = 1:size(times, 2)
    timeStep = times(i);
    a = strcat('dischargeNorth', num2str(times(i)), 'sV5Dot6.mat');
    discharge = load(a);
    discharge = discharge.discharge;
    if timeStep == 1
        plot(discharge, 'LineWidth', 2)
    else
        apprDischarge = arrayfun(@(i) mean(discharge(i:i+timeStep-1)),1:timeStep:length(discharge)-timeStep+1)'; % the averaged vector
        c = (timeStep/2):timeStep:numel(discharge);
        c = c(1:end-1);
        plot(c, apprDischarge, 'LineWidth', 2)
        
        
        % Get error estimates
        % L2-norm
        normErrors(i) = sqrt(sum(abs(d1.discharge(c) - apprDischarge))) * sqrt(timeStep);
        
        %normErrors(i-1) = norm(apprDischarge - d1.discharge(c))/norm(d1.discharge(c));
        
        %normErrors(i-1) = sum(abs(apprDischarge - d1.discharge(c)))/size(c, 2);
        %errorsMax(i-1) = max(abs(apprDischarge - d1.discharge(c)));
        %maxAtTime(i-1) = c(find(abs(apprDischarge - d1.discharge(c)) == errorsMax(i-1)));
    end
end
titles = strtrim(cellstr(num2str(times'))');
legend(titles);
end
