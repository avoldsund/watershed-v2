function [faces, faceNormals, sign] = flipNormalsOutwards(CG, cellIx)
%FLIPNORMALSOUTWARDS flips a cell's normals outwards
%   [FACES, FACENORMALS] = FLIPNORMALSOUTWARDS(CG, CELLIX) flips the
%   normals of cell CELLIX. CELLIX's face indices FACES are returned, as
%   well as their FACENORMALS.

startIx = CG.cells.facePos(cellIx);
endIx = CG.cells.facePos(cellIx + 1) - 1;
faces = CG.cells.faces(startIx:endIx, 1);
faceNormals = CG.faces.normals(faces, :);
sign = 1 - 2 * (CG.faces.neighbors(faces, 1) ~= rldecode(cellIx, size(faces, 1)));
faceNormals(sign == -1, :) = -faceNormals(sign == -1, :);

end