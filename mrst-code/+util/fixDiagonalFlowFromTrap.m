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
nbrs = nbrs(nbrs ~= trapCellIx);

% If any of the neighbors are at the boundary, return
if any(nbrs == 0)
    return
end

% The trap spills diagonally to two cells, over faces spFaces
for ix = 1:size(spFaces, 2)
    % Find outflow faces from the neighbor
    [faces, nrmls] = util.flipNormalsOutwards(CG, nbrs(ix));
    dp = sum(bsxfun(@times, nrmls, CG.cells.fd(nbrs(ix), :)), 2);
    posIndices = find(dp > 0);

    % If one of the faces the neighbor has flow out of is the incoming 
    % diagonal (spill face), we change the flow direction of the neighbor's
    % faces. This is to avoid a potential zero flow over the edge.
    % NB: Remember to multiply by sqrt(2) to get a new normalized fd.
    if faces(posIndices) == spFaces(1, ix)
        %newFlowDir =  CG.cells.fd(trapCellIx, :) .* sqrt(2) + CG.cells.fd(nbrs(ix), :);
        newFlowDir =  CG.cells.fd(trapCellIx, :);
        %assert(sqrt(newFlowDir(1)^2+newFlowDir(2)^2) == 1)
        faceIndices = CG.cells.facePos(nbrs(ix)):CG.cells.facePos(nbrs(ix) + 1) - 1;    
        faceFlowDirections(faceIndices, :) = rldecode(newFlowDir, size(faceIndices, 2));
    end
end