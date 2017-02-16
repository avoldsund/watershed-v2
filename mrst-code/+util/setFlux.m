function flux = setFlux(CG, nrOfTraps)
%SETFLUX Summary of this function goes here
%   Detailed explanation goes here

% Calculate flux for notTrapCells
N = CG.cells.num - nrOfTraps;


% Dot product of face normals and flow direction vectors
faceIndices = CG.cells.faces(1:N*4)';
faceNormals = CG.faces.normals(faceIndices, :);

a = repelem(CG.cells.fd(1:N, 1), 4);
b = repelem(CG.cells.fd(1:N, 2), 4);
c = horzcat(a, b);

dotProduct = sum(faceNormals .* c, 2);


flux = zeros(CG.faces.num, 1);
flux(1:N) = dotProduct;

end

