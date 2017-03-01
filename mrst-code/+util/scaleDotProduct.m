function [dotProduct] = scaleDotProduct(CG, dotProduct)
%SCALEDOTPRODUCT Summary of this function goes here
%   Detailed explanation goes here
    
    N = size(CG.cells.faces, 1);
    faceIndices = CG.cells.faces(:, 1);
    nbrsOfFaces = CG.faces.neighbors(faceIndices, :);
    bothNbrsInInterior = all(nbrsOfFaces ~= 0, 2);
    validIndices = nbrsOfFaces(bothNbrsInInterior, :);
    deltaZ = zeros(N, 1);
    deltaZ(bothNbrsInInterior) = abs(CG.cells.z(validIndices(:, 1)) - CG.cells.z(validIndices(:, 2)));
    
    % Get indices where deltaZ is zero
    indicesOfZeroDeltaZ = all(deltaZ == 0, 2);
    deltaZ(indicesOfZeroDeltaZ) = 0.1;
    dotProduct = dotProduct .* deltaZ;
    
    % Divide by distance
    dotProduct = dotProduct * 0.1;
end

