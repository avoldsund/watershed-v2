function faces = getCellFaces(CG, cellIx)
%GETCELLFACES returns the indices of a cell
%   FACES = GETCELLFACES(CG, CELLIX) returns the face indices of cell
%   CELLIX. Simple help method.

startIx = CG.cells.facePos(cellIx);
endIx = CG.cells.facePos(cellIx + 1) - 1;
faces = CG.cells.faces(startIx:endIx, 1);

end