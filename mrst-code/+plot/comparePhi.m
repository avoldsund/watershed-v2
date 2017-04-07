function [] = comparePhi(phiOne, phiTwo)
    %COMPAREPHI Compares the time-of-flight for two different porosities
    %   COMPAREPHI(PHIONE, PHITWO) returns the figures that shows the
    %   time-of-flight for one calculation with porosity PHIONE, and
    %   another with porosity PHITWO.
    
    scale = true;
    [CG, tofOne] = calculateTof(phiOne, scale);
    [~, tofTwo] = calculateTof(phiTwo, scale);
    showTofCentroids = true;
    saveNameOne = false;
    saveNameTwo = false;
    
    [fOne, hOne] = plot.tof(CG, tofOne, showTofCentroids, saveNameOne);
    caxis([0, max(tofOne)])
    [fTwo, hTwo] = plot.tof(CG, tofTwo, showTofCentroids, saveNameTwo);
    caxis([0, max(tofOne)])
    
end