function [] = compareScaling(phi)
    %COMPARESCALING Compares the time-of-flights for where one has its
    %fluxes scaled by derivatives, and the other assumes a similar delta Z
    %over flux faces
    % [] = COMPARESCALING(PHI) calculates the time-of-flight for two
    % scenarios where one scenario uses height differences to calculate the
    % fluxes, and the other does not. PHI is the porosity of the trap
    % cells.
    
    [CG, tofScale] = calculateTof(phi, true);
    [~, tofNoScale] = calculateTof(phi, false);
    
    showTofCentroids = true;
    saveNameOne = 'derivativeScaling';
    saveNameTwo = 'noDerivativeScaling';
    
    [fOne, hOne] = plot.tof(CG, tofScale, showTofCentroids, saveNameOne);
    [fTwo, hTwo] = plot.tof(CG, tofNoScale, showTofCentroids, saveNameTwo);
    
end

