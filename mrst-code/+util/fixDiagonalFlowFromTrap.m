function [faceFlowDirections] = fixDiagonalFlowFromTrap(CG, spFaces, trapCellIx, faceFlowDirections)
%FIXDIAGONALFLOWFROMTRAP Summary of this function goes here
%   Detailed explanation goes here

faceOne = spFaces(1, 1);
faceTwo = spFaces(1, 2);

nbrsOne = CG.faces.neighbors(faceOne, :);
nbrsTwo = CG.faces.neighbors(faceTwo, :);
nbrIxOne = nbrsOne ~= trapCellIx;
nbrIxTwo = nbrsTwo ~= trapCellIx;
nbrOne = nbrsOne(nbrIxOne);
nbrTwo = nbrsTwo(nbrIxTwo);

count = 0;
if nbrOne == 0 || nbrTwo == 0
    trapCellIx
    count = count + 1
    return
end

[facesOne, nrmlsOne] = util.flipNormalsOutwards(CG, nbrOne);
[facesTwo, nrmlsTwo] = util.flipNormalsOutwards(CG, nbrTwo);
dpOne = sum(bsxfun(@times, nrmlsOne, CG.cells.fd(nbrOne, :)), 2);
dpTwo = sum(bsxfun(@times, nrmlsTwo, CG.cells.fd(nbrTwo, :)), 2);
posIndicesOne = find(dpOne > 0);
posIndicesTwo = find(dpTwo > 0);

% If the only outflow face is similar to the incoming diagonal, change fd
% of nbrCell
if facesOne(posIndicesOne) == faceOne
    newFlowDir =  CG.cells.fd(trapCellIx, :) + CG.cells.fd(nbrOne, :);
    faceIndices = CG.cells.facePos(nbrOne):CG.cells.facePos(nbrOne + 1) - 1;
    
    faceFlowDirections(faceIndices, :) = rldecode(newFlowDir, size(faceIndices, 2));
    
end
if facesTwo(posIndicesTwo) == faceTwo
    newFlowDir =  CG.cells.fd(trapCellIx, :) + CG.cells.fd(nbrTwo, :);
    faceIndices = CG.cells.facePos(nbrTwo):CG.cells.facePos(nbrTwo + 1) - 1;
    faceFlowDirections(faceIndices, :) = rldecode(newFlowDir, size(faceIndices, 2));
end

end

