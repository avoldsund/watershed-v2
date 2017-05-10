function discharge = hydrographUniform(CG, tof, amount, duration)
%CALCULATEHYDROGRAPH Calculates a hydrograph for a precipitation event
%where the precipitation is uniform in the entire landscape.
%   DISCHARGE = HYDROGRAPHUNIFORM(CG, TOF, INTENSITY, DURATION) returns the
%   discharge over time given the time-of-flight TOF, the intensity of the
%   precipitation INTENSITY and the time it lasts DURATION.

finalTime = max(tof) + duration + 1;
duration = ones(CG.cells.num, 1) .* duration;
amount = ones(CG.cells.num, 1) .* amount .* CG.cells.volumes;
discharge = zeros(finalTime, 1);

for time = 0:finalTime

    % Cells contributing to flow between time [i, i+1)
    contributingCells = tof <= time & tof + duration > time;
    
    if size(amount(contributingCells), 1) > 0
        flowInTimeStep = sum(amount(contributingCells));
        discharge(time+1) = flowInTimeStep;
    end
end

discharge = discharge * (10^-3) / 3600;

end