function flux = setFlux2(CG)
% This function is NOT in use

% Make all face normals point outwards from a cell
faces = CG.cells.faces(:, 1);
faceNormals = CG.faces.normals(faces, :);
sign = 1 - 2 * (CG.faces.neighbors(faces, 1) ~= rldecode((1:CG.cells.num)', diff(CG.cells.facePos)));
faceNormals(sign == -1, :) = -faceNormals(sign == -1, :);

flowDirections = rldecode(CG.cells.fd, diff(CG.cells.facePos));
dotProduct = sum(faceNormals .* flowDirections, 2);
dotProduct(dotProduct < 0) = 0;

% boundary = find(CG.faces.neighbors(:, 1) == 0);

% Get the fluxes goint out of cells
faceIxWithFlow = find(dotProduct > 0);
facesWithFlow = faces(faceIxWithFlow);
nrmls = CG.faces.normals(facesWithFlow, :);
flwdir = flowDirections(faceIxWithFlow, :);
flux = zeros(CG.faces.num, 1);
dp = sum(nrmls .* flwdir, 2);

% Remove conflicting indices
a = sortrows(facesWithFlow);
ambiguousFacesIx = find(diff(a) == 0);
ambiguousFaces = a(ambiguousFacesIx);

temp = ambiguousFacesIx + 1;
removeIx = vertcat(ambiguousFacesIx, temp);
faceIxWithFlow(removeIx) = [];
dp(removeIx) = [];
facesWithFlow = faces(faceIxWithFlow);

flux(facesWithFlow) = dp;


end