function volume = getVolume(A, R)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here
    vol = @(A, R) 2/9 * pi * R^2 * A * (1 - exp(-9/2));    
    volume = vol(A, R);
    
end

