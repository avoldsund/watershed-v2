function [flux, faceFlowDirections] = setFlux(CG, nrOfTraps, outletTrapNr, scale)
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

% Why faces without flow directions??? Faces with neighbor out of domain
% boundary (cell 0)

% Correct face flow directions for cells with one or two faces
for i = 1:size(facesPerCell, 1) - nrOfTraps
    if facesPerCell(i) == 2 || facesPerCell(i) == 1
        
        interval = CG.cells.facePos(i):CG.cells.facePos(i+1)-1;
        faceIx = CG.cells.faces(interval);
        nbrs = CG.faces.neighbors(faceIx, :);
        chosenIx = nbrs(find(nbrs ~= i)) > N; % Get trap cell nbr
        chosenFace = faceIx(chosenIx); % Get cell facing towards trap
        
        if facesPerCell(i) == 1
            faceNormals(interval(chosenIx), :) = CG.cells.fd(i, :) * CG.faceLength;
        else
            faceFlowDirections(interval(chosenIx), :) = bsxfun(@rdivide, CG.faces.normals(chosenFace, :), sqrt(sum(CG.faces.normals(chosenFace, :).^2, 2)));
        end
    end
end

% Possibly change face normals for cells where face normal is [0, 0].
% Only happens when traps are involved and a cell has three faces.
fIx = find(CG.faces.normals(:, 1) == 0 & CG.faces.normals(:, 2) == 0);
faceCoords = CG.faces.centroids(fIx, :);
for i = 1:size(fIx, 1)
    cIx = find(CG.cells.centroids(:, 1) == faceCoords(i, 1) & ...
               CG.cells.centroids(:, 2) == faceCoords(i, 2));
    interval = CG.cells.facePos(cIx):CG.cells.facePos(cIx + 1) - 1;
    [~, fNrmls, ~] = util.flipNormalsOutwards(CG, cIx);
    d = sum(fNrmls .* faceFlowDirections(interval, :), 2);
    ix = find(fNrmls(:, 1) == 0 & fNrmls(:, 2) == 0);
    if any(d > 0) == 0 % Check for any outflow
        % Remember to scale faceNormal by face area
        faceNormals(interval(ix), :) = CG.cells.fd(cIx, :) * CG.faceLength; 
    end
end

% Change faceFlowDirection for spill pair face to ensure flow out of trap
for i = 1:nrOfTraps
    if i == outletTrapNr
        continue 
    end
    trapCellIx = CG.cells.num - nrOfTraps + i;
    if trapCellIx == 139015
       disp('heeeey') 
    end
    [spFaces, indices] = util.getSpillPointFace(CG, nrOfTraps, i);
    
    if size(indices, 2) > 1
        faceFlowDirections = util.fixDiagonalFlowFromTrap(CG, spFaces, trapCellIx, faceFlowDirections);
        faceFlowDirections(indices, :) = repelem(CG.cells.fd(trapCellIx, :), size(spFaces, 2), 1);
    else
        nbrCell = CG.faces.neighbors(spFaces, :);
        nbrCell = nbrCell(nbrCell ~= trapCellIx);
        
        % This should be more accurate than the line below, as the
        % faceFlowDirections and faceNormals have been changed for a reason
        % during the algorithm
        dp = sum((faceNormals(indices, :) ~= 0) .* faceFlowDirections(indices, :));
        %dp = sum((CG.faces.normals(spFaces, :) ~= 0) .* CG.cells.fd(nbrCell, :));
        if dp == 0 % dotProduct of faceNbr is zero, set faceFlowDir to faceNormal
            % Flip the faceNormal as they point inward. No flip if nbrCell
            % is trapCell
            if nbrCell > CG.cells.num - nrOfTraps
                faceFlowDirections(indices, :) = bsxfun(@rdivide, CG.faces.normals(spFaces, :), sqrt(sum(CG.faces.normals(spFaces, :).^2, 2)));
            else
                faceFlowDirections(indices, :) = bsxfun(@rdivide, -CG.faces.normals(spFaces, :), sqrt(sum(CG.faces.normals(spFaces, :).^2, 2)));
            end
        end
    end
end

flux = util.calculateFlux(CG, faceNormals, faceFlowDirections, scale);

% Do the average of fluxes
flux = util.averageFluxes(faceIndices, flux);

end