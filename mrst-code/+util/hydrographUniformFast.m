function discharge = hydrographUniformFast(CG, tof, amount, duration)
%CALCULATEHYDROGRAPH Calculates a hydrograph for a precipitation event
%where the precipitation is uniform in the entire landscape.
%   DISCHARGE = HYDROGRAPHUNIFORM(CG, TOF, INTENSITY, DURATION) returns the
%   discharge over time given the time-of-flight TOF, the intensity of the
%   precipitation INTENSITY and the time it lasts DURATION.

discharge = zeros(max(tof) + duration + 1, 1);

for i = 1:CG.cells.num
    times = tof(i) + 1 : tof(i) + duration;
    discharge(times) = discharge(times) + CG.cells.volumes(i) * amount;
end

discharge = discharge * (10^-3) / 3600;

end