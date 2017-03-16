function [spFaces, indices] = getSpillPointFace(CG, nrOfTraps, trapNr)
%GETSPILLPOINTFACES Calculate closest faces to spill point in trap
%   [SPFACES, INDICES] = GETSPILLPOINTFACE(CG, NROFTRAPS, TRAPNR) returns
%   the faces closest to the spill point in trap TRAPNR. The unique face
%   numbers SPFACES and their indices in CG.cells.faces INDICES are
%   returned.

spCoord = CG.parent.cells.centroids(CG.spillPoints(trapNr), :);

% Get faces of trap
trapCellIx = CG.cells.num - nrOfTraps + trapNr;
startIx = CG.cells.facePos(trapCellIx);
endIx = CG.cells.facePos(trapCellIx + 1) - 1;
trapFaces = CG.cells.faces(startIx:endIx);

% Get trap and spill point centroids. Perturb spill point centroid in flow
% direction
trapCentroids = CG.faces.centroids(trapFaces, :);
flowDirFromSpillPoint = CG.cells.fd(trapCellIx, :);
spCoord = bsxfun(@plus, spCoord, flowDirFromSpillPoint);

% Calculate distance from trap faces to perturbed centroid. Find the
% face closest to the spill point
distance = util.calculateEuclideanDist(trapCentroids, spCoord);
faceIx = find(distance == min(distance(:)));

spFaces = trapFaces(faceIx);
interval = startIx:endIx;
indices = interval(faceIx);

end

