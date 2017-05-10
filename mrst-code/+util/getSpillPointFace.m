function [spFaces, indices] = getSpillPointFace(CG, nrOfTraps, trapNr)
%GETSPILLPOINTFACES Calculate closest faces to spill point in trap
%   [SPFACES, INDICES] = GETSPILLPOINTFACE(CG, NROFTRAPS, TRAPNR) returns
%   the faces closest to the spill point in trap TRAPNR. The unique face
%   numbers SPFACES and their indices in CG.cells.faces INDICES are
%   returned.

spCentroid = CG.parent.cells.centroids(CG.spillPoints(trapNr), :);

% Get faces of trap
trapCellIx = CG.cells.num - nrOfTraps + trapNr;
if trapCellIx == 77408
   disp('abc') 
end
startIx = CG.cells.facePos(trapCellIx);
endIx = CG.cells.facePos(trapCellIx + 1) - 1;
trapFaces = CG.cells.faces(startIx:endIx);

% Get trap and spill point centroids.
trapCentroids = CG.faces.centroids(trapFaces, :);
flowDirFromSpillPoint = CG.cells.fd(trapCellIx, :);

% % Method #1
% Calculate distance from trap faces to perturbed centroid. Find the
% face closest to the spill point. Perturb more than 1 * flowDirFromSpillPoint 
% to avoid corner cases, specifically one where an L-shaped closest face is
% not chosen (centroid in weird position).
% spCoord = bsxfun(@plus, spCentroid, 4 * flowDirFromSpillPoint); 
% distance = util.calculateEuclideanDist(trapCentroids, spCoord);
% errTol = 10^-10;
% faceIx = find(distance < min(distance) + errTol); 
% 
% spFaces = trapFaces(faceIx);
% interval = startIx:endIx;
% indices = interval(faceIx);

% Method #2
vecToFaceCentroids = bsxfun(@minus, trapCentroids, spCentroid);
lenFlowDir = sqrt(flowDirFromSpillPoint(1)^2 + flowDirFromSpillPoint(2)^2);
lenVecToFaceCentroids = sqrt(vecToFaceCentroids(:, 1).^2 + vecToFaceCentroids(:, 2).^2);

cosTheta = sum(bsxfun(@times, vecToFaceCentroids, flowDirFromSpillPoint), 2) ...
    ./ bsxfun(@times, lenVecToFaceCentroids, lenFlowDir);
errTol = 10^-10;
faceIx = find(cosTheta ./ lenVecToFaceCentroids > (max(cosTheta ./ lenVecToFaceCentroids - errTol)));

spFaces = trapFaces(faceIx);
interval = startIx:endIx;
indices = interval(faceIx);

end
