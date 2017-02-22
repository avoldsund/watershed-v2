function [spFaces, indices] = getSpillPointFace(CG, nrOfTraps, trapNr)
%GETSPILLPOINTFACES Summary of this function goes here
%   Detailed explanation goes here

spCoord = CG.parent.cells.centroids(CG.spillPoints(trapNr), :);

% Get faces of trap
trapCellIx = CG.cells.num - nrOfTraps + trapNr;
startIx = CG.cells.facePos(trapCellIx);
endIx = CG.cells.facePos(trapCellIx + 1) - 1;
trapFaces = CG.cells.faces(startIx:endIx);

% Get trap and spill point centroids. Perturb sp centroid in flow
% direction
trapCentroids = CG.faces.centroids(trapFaces, :);
flowDirFromSpillPoint = CG.cells.fd(trapCellIx, :);
spCoord = bsxfun(@plus, spCoord, flowDirFromSpillPoint);

% Calculate distance from trap faces to perturbed sp centroid
distance = util.calculateEuclideanDist(trapCentroids, spCoord);
%dFromSp = abs(sum(bsxfun(@minus, trapCentroids, spCoord), 2));
faceIx = find(distance == min(distance(:)));

spFaces = trapFaces(faceIx);
interval = startIx:endIx;
indices = interval(faceIx);

end

