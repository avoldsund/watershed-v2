function faceFlowDirections = fixDiagonalFlowFromTrap(CG, spFaces, trapCellIx, faceFlowDirections)
%FIXDIAGONALFLOWFROMTRAP looks at spill points with diagonal flow to ensure
%continuation of flow.
%   FACEFLOWDIRECTIONS = FIXDIAGONALFLOWFROMTRAP(CG, SPFACES, TRAPCELLIX,
%   FACEFLOWDIRECTIONS) might alter the flow directions of the spill points
%   neighbor cells if the spill point's flow direction leads to the
%   neighbor cell not having outflow. SPFACES is the unique indices of the
%   faces closest to the spill point in trap TRAPCELLIX. FACEFLOWDIRECTIONS
%   might get updated for the neighbors.

% Get neighbors of the two faces
nbrs = CG.faces.neighbors(spFaces, :);
notTraps = nbrs ~= trapCellIx;
faceNbrs = nbrs(notTraps);

% If any of the neighbors are at the boundary, return
if any(faceNbrs == 0)
    return
end

for cIx = 1:2
    % Find outflow faces from the two neighbors
    [faces, nrmls] = util.flipNormalsOutwards(CG, faceNbrs(cIx));
    dp = sum(bsxfun(@times, nrmls, CG.cells.fd(faceNbrs(cIx), :)), 2);
    posIndices = find(dp > 0);

    % If the only outflow face is similar to the incoming diagonal, change fd
    % of nbrCell
    if faces(posIndices) == spFaces(1, cIx)
        newFlowDir =  CG.cells.fd(trapCellIx, :) + CG.cells.fd(faceNbrs(cIx), :);
        faceIndices = CG.cells.facePos(faceNbrs(cIx)):CG.cells.facePos(faceNbrs(cIx) + 1) - 1;    
        faceFlowDirections(faceIndices, :) = rldecode(newFlowDir, size(faceIndices, 2));
    end
end

