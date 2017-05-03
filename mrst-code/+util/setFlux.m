function [flux, faceFlowDirections] = setFlux(CG, nrOfTraps, outletTrapNr, scale)
%SETFLUX returns the fluxes for all faces in the coarse grid
%   [FLUX, FACEFLOWDIRECTIONS] = SETFLUX(CG, NROFTRAPS, OUTLETTRAPNR,
%   SCALE) calculates the fluxes FLUX for all faces in the coarse grid CG.
%   The flow directions of the faces FACEFLOWDIRECTIONS are also returned.
%   SCALE tells whether the fluxes shall be scaled by the elevations in the
%   landscape.

% Calculate flux for notTrapCells
N = CG.cells.num - nrOfTraps;

% Dot product of face normals and flow direction vectors
faceIndices = CG.cells.faces(:, 1);
faceNormals = CG.faces.normals(faceIndices, :);
facesPerCell = CG.cells.facePos(2:end) - CG.cells.facePos(1:end-1);

% Set flow directions for all faces not in a trap
nRegularFaces = sum(facesPerCell(1:CG.cells.num - nrOfTraps));
allCellIndices = repelem(1:CG.cells.num, diff(CG.cells.facePos))';
faceFlowDirections = zeros(sum(facesPerCell), 2);
faceFlowDirections(1:nRegularFaces, :) = CG.cells.fd(allCellIndices(1:nRegularFaces), :);

% Set flow directions for all faces in traps. As nbr cells at the outside
% of the domain has no flow direction, we remove nbrs that are 0.
trapFaces = faceIndices(nRegularFaces + 1:end);
tNbrs = CG.faces.neighbors(trapFaces, 1);
validNbrs = tNbrs ~= 0;
interval = (nRegularFaces + 1:size(CG.cells.faces, 1))';
trapNbrs = tNbrs(validNbrs);
faceFlowDirections(interval(validNbrs), :) = CG.cells.fd(trapNbrs, :);

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

% Change faceFlowDirection for spill pair faces to ensure outflow from trap
for i = 1:nrOfTraps
    % Do not alter the faces in the trap if the watershed outlet is a part
    % of it
    if i == outletTrapNr
        continue 
    end
    
    % Retrieve spill faces in each trap and set the face flow directions to
    % be the flow direction of the outlet.
    trapCellIx = CG.cells.num - nrOfTraps + i;
    [spFaces, indices] = util.getSpillPointFace(CG, nrOfTraps, i);
    faceFlowDirections(indices, :) = repelem(CG.cells.fd(trapCellIx, :), size(spFaces, 2), 1);
    
    % Make sure that the neighbors of the spill faces do not cancel out the
    % flow from the trap.
    nbrs = CG.faces.neighbors(spFaces, :);
    nbrs = nbrs(nbrs ~= trapCellIx);
    % If any of the neighbors are at the boundary, return
    if any(nbrs == 0)
        continue
    end
    
    for ix = 1:size(spFaces, 2)
        % Find outflow faces from the neighbor
        [faces, nrmls] = util.flipNormalsOutwards(CG, nbrs(ix));
        nbrFaces = CG.cells.facePos(nbrs(ix)):CG.cells.facePos(nbrs(ix)+1)-1;
        dp = sum(bsxfun(@times, nrmls, faceFlowDirections(nbrFaces, :)), 2);
        
        % If the nbr only has outflow towards the trap, change its face
        % flow directions
        posIndices = find(dp > 0);
        if faces(posIndices) == spFaces(1, ix)% & size(posIndices, 1) == 1
            newFlowDir = CG.cells.fd(trapCellIx, :) .* sqrt(2) + CG.cells.fd(nbrs(ix), :);
            %assert(sqrt(newFlowDir(1)^2+newFlowDir(2)^2) == 1)
            faceFlowDirections(nbrFaces, :) = rldecode(newFlowDir, size(nbrFaces, 2));
            
            % If one of the nbr's faces is a trap cell, change the face
            % flow direction of the common face
            nbrsOfFaces = CG.faces.neighbors(faces, :);
            % nTrapCells = unique(nbrsOfFaces(nbrsOfFaces ~= trapCellIx & nbrsOfFaces > CG.cells.num-nrOfTraps  & nbrsOfFaces ~= faces));
            nTrapCells = unique(nbrsOfFaces(nbrsOfFaces ~= trapCellIx & nbrsOfFaces > CG.cells.num-nrOfTraps  & ~ismember(nbrsOfFaces, faces)));
            
            for j = 1:size(nTrapCells, 1)
                tCell = nTrapCells(j);
                tFaces = util.getCellFaces(CG, tCell);
                tFaceIndices = CG.cells.facePos(tCell):CG.cells.facePos(tCell + 1) - 1;
                commonIx = ismember(tFaces, faces);
                indices = tFaceIndices(commonIx);
                if size(indices, 2) > 1
                   disp('wwwwwaaaaat') 
                end
                faceFlowDirections(indices, :) = rldecode(CG.cells.fd(trapCellIx, :), size(indices, 2));
            end
        end
    end
end

flux = util.calculateFlux(CG, faceNormals, faceFlowDirections, scale);
flux = util.averageFluxes(faceIndices, flux);

% for i = 1:CG.cells.num
%     faces = util.getCellFaces(CG, i);
%     nbrs = CG.faces.neighbors(faces, :);
%     if sum(any(nbrs ~= 0 & nbrs ~= i)) > 0
%         continue
%     else
%         flux(faces) = 0;
%     end
% end

% for i = 1:CG.cells.num
%     faces = util.getCellFaces(CG, i);
%     nbrs = CG.faces.neighbors(faces, :);
%     % Identify which faces have outflow and inflow
%     faces = util.getCellFaces(CG, i);
%     [faceFlipped, ~, sign] = util.flipNormalsOutwards(CG, i);
%     faceFluxes = sign .* flux(faces);
%     if all(faceFluxes <= 0) & sum(any(nbrs ~= 0 & nbrs ~= i)) > 0 & i ~= 77355
%         if i > CG.cells.num - nrOfTraps % it is a trap
%             [spFaces, indices] = util.getSpillPointFace(CG, nrOfTraps, i-(CG.cells.num - nrOfTraps));
%             flux(spFaces) = 0.1 * sign(ismember(spFaces, faceFlipped));
%         else
%             disp('noTrap')
%         end
%         
%     end
% end

end