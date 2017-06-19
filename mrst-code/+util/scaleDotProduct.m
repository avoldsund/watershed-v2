function dotProduct = scaleDotProduct(CG, dotProduct)
%SCALEDOTPRODUCT Scale the flux based on slope or other parameters
%   DOTPRODUCT = SCALEDOTPRODUCT(CG, DOTPRODUCT) scales the flux based on
%   different parameters such as slope, and returns the new DOTPRODUCT.
    
    %N = size(CG.cells.faces, 1);
    %faceIndices = CG.cells.faces(:, 1);
    %nbrsOfFaces = CG.faces.neighbors(faceIndices, :);
    %bothNbrsInInterior = all(nbrsOfFaces ~= 0, 2);
    %validIndices = nbrsOfFaces(bothNbrsInInterior, :);
    %deltaZ = zeros(N, 1);
    for i = 1:CG.cells.num
        faces = util.getCellFaces(CG, i)
        
        dotProduct()
    end
    
    % Absolute value of slope 
    %deltaZ(bothNbrsInInterior) = abs(CG.cells.z(validIndices(:, 1)) - CG.cells.z(validIndices(:, 2)));
    
    % Negative slopes set to constant
    %deltaZ(bothNbrsInInterior) = CG.cells.z(validIndices(:, 1)) - CG.cells.z(validIndices(:, 2));
    %uphillSlope = find(deltaZ < 0);
    %deltaZ(uphillSlope) = 0.2;
    
    % Get indices where deltaZ is zero
    %indicesOfZeroDeltaZ = all(deltaZ == 0, 2);
    %deltaZ(indicesOfZeroDeltaZ) = 0.1;
    %dotProduct = dotProduct .* deltaZ;
    
    % Divide by distance
    dotProduct = dotProduct * 0.1;
end