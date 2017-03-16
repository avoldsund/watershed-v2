function faceFlowDirections = fixDiagonalFlowFromTrap(CG, spFaces, trapCellIx, faceFlowDirections)
%FIXDIAGONALFLOWFROMTRAP looks at spill points with diagonal flow to ensure
%continuation of flow.
%   FACEFLOWDIRECTIONS = FIXDIAGONALFLOWFROMTRAP(CG, SPFACES, TRAPCELLIX,
%   FACEFLOWDIRECTIONS) might alter the flow directions of the spill points
%   neighbor cells if the spill point's flow direction leads to the
%   neighbor cell not having outflow. SPFACES is the unique indices of the
%   faces closest to the spill point in trap TRAPCELLIX. FACEFLOWDIRECTIONS
%   might get updated for the neighbors.

faceOne = spFaces(1, 1);
faceTwo = spFaces(1, 2);

% Get neighbors of the two faces
nbrs = CG.faces.neighbors(spFaces, :);
notTraps = nbrs ~= trapCellIx;
faceNbrs = nbrs(notTraps);
nbrOne = faceNbrs(1);
nbrTwo = faceNbrs(2);

% If any of the neighbors are at the boundary, return
if any(faceNbrs == 0)
    return
end

% Find outflow faces from the two neighbors
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

