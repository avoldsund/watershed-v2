function faces = getCellFaces(CG, cellIx)
%GETCELLFACES Summary of this function goes here
%   Detailed explanation goes here

startIx = CG.cells.facePos(cellIx);
endIx = CG.cells.facePos(cellIx + 1) - 1;

faces = CG.cells.faces(startIx:endIx, 1);

end