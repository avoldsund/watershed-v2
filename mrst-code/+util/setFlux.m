function flux = setFlux(CG, nrOfTraps)
%SETFLUX Summary of this function goes here
%   Detailed explanation goes here

% Calculate flux for notTrapCells
N = CG.cells.num - nrOfTraps;

% Dot product of face normals and flow direction vectors
faceIndices = CG.cells.faces(:, 1);
faceNormals = CG.faces.normals(faceIndices, :);


facesPerCell = CG.cells.facePos(2:end) - CG.cells.facePos(1:end-1);
faceFlowDirections = zeros(sum(facesPerCell), 2);

% Set flow directions for all faces not in a trap
for i = 1:size(facesPerCell, 1) - nrOfTraps
    faceFlowDirections(CG.cells.facePos(i):CG.cells.facePos(i+1)-1, :) = repelem(CG.cells.fd(i, :), facesPerCell(i), 1);
end

% Set flow directions for all faces in traps
for i = 1:nrOfTraps
    startIx = CG.cells.facePos(N+i);
    endIx = CG.cells.facePos(N+i+1)-1;
    faces = CG.cells.faces(startIx:endIx);
    nbrCells = CG.faces.neighbors(faces);
    
    % Valid interval removes flow directions to cells out of boundary
    validIx = find(nbrCells);
    validNbrs = nbrCells(validIx);
    interval = startIx:endIx;
    validInterval = interval(validIx);
    
    flowDirOnFaces = CG.cells.fd(validNbrs, :);
    faceFlowDirections(validInterval, :) = flowDirOnFaces;
end

% Correct face flow directions for cells with only two faces

for i = 1:size(facesPerCell, 1) - nrOfTraps
    if facesPerCell(i) == 2
        interval = CG.cells.facePos(i):CG.cells.facePos(i+1)-1;
        faceIx = CG.cells.faces(interval);
        nbrs = CG.faces.neighbors(faceIx, :);
        chosenIx = nbrs(find(nbrs ~= i)) > N;
        chosenFace = faceIx(chosenIx);

        faceFlowDirections(interval(chosenIx), :) = CG.faces.normals(chosenFace, :);
    end
end

% Change faceFlowDirection for spill pair face to ensure flow out of trap
for i = 1:nrOfTraps
   trapCellIx = CG.cells.num - nrOfTraps + i;
   [spFaces, indices] = util.getSpillPointFace(CG, nrOfTraps, i);
   faceFlowDirections(indices, :) = repelem(CG.cells.fd(trapCellIx, :), size(spFaces, 2), 1);
end

dotProduct = sum(faceNormals .* faceFlowDirections, 2);

% Do the average of fluxes
a = horzcat(faceIndices, dotProduct);
b = sortrows(a, 1);  % Sort rows based on face indices
averageOfFluxes = accumarray(b(:,1), b(:,2)) ./ accumarray(b(:,1), 1);

flux = averageOfFluxes;

end
