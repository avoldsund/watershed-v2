function mapFacesToCells = mapFacesToCells(CG, nrOfTraps)
%MAPFACESTOCELLS Summary of this function goes here
%   Detailed explanation goes here

N = size(CG.cells.faces, 1);
nCells = CG.cells.num;
T = nCells - nrOfTraps;
facesPerCell = CG.cells.facePos(2:end) - CG.cells.facePos(1:end-1);
notTrapFaces = sum(facesPerCell(1:T));

mapFacesToCells = zeros(N, 1);
mapFacesToCells(CG.cells.faces(1:notTrapFaces)) = repelem(1:T, facesPerCell(1:T));
mapFacesToCells(CG.cells.faces(notTrapFaces+1:end)) = repelem(T+1:nCells, facesPerCell(T+1:end));

end