function [faces, faceNormals] = flipNormalsOutwards(CG, cellIx)
%FLIPNORMALSOUTWARDS Summary of this function goes here
%   Detailed explanation goes here

startIx = CG.cells.facePos(cellIx);
endIx = CG.cells.facePos(cellIx + 1) - 1;
faces = CG.cells.faces(startIx:endIx, 1);
faceNormals = CG.faces.normals(faces, :);
sign = 1 - 2 * (CG.faces.neighbors(faces, 1) ~= rldecode(cellIx, size(faces, 1)));
faceNormals(sign == -1, :) = -faceNormals(sign == -1, :);

end