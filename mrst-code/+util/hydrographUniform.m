function flow = calculateHydrographUniform(CG, tof, scale, amount)
%CALCULATEHYDROGRAPH Summary of this function goes here
%   Detailed explanation goes here

% Get tof in minutes or hours and round up
tof = floor(tof ./ scale);

% Get a duration array and a rain amount array

% Amount is given in mm/hour
%if scale == 60
%    amount = amount/60;
%end

duration = 5;
[duration, amount] = uniformPrecipitation(CG, duration, amount);

% Multiply amount by volume (area)
amount = amount .* CG.cells.volumes;

% Duration is in the same time scale as tof

% Iterate over time
totalTime = 2500;
flow = zeros(totalTime, 1);

for time = 0:totalTime
    
    % Cells contributing to flow between time i and i+1
    contributingCells = tof <= time & tof + duration > time;
    
    if size(amount(contributingCells), 1) > 0
        flowInTimeStep = sum(amount(contributingCells));
        flow(time+1) = flowInTimeStep;
    end
end

flow = flow * (10^-3)/3600;

end

function [duration, amount] = uniformPrecipitation(CG, duration, amount)

% startTime = ones(CG.cells.num, 1) .* startTime;
duration = ones(CG.cells.num, 1) .* duration;
amount = ones(CG.cells.num, 1) .* amount;

end