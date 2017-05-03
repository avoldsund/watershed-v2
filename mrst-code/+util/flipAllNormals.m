function [allCellFaces, allFaceNormals, allNbrs, nbrPairs, signs] = flipAllNormals(CG)
    %FLIPALLNORMALS returns all cell faces, all face normals flipped
    %outwards and the signs: -1 if flipped, 1 if not
    %   [ALLCELLFACES, ALLFACENORMALS, SIGN] = FLIPALLNORMALS(CG) takes a
    %   grid CG and returns all cell faces in the grid ALLCELLFACES, all
    %   face normals flipped outwards ALLFACENORMALS and the signs SIGNS
    %   which says if the face normal has been flipped or not.

allCellIndices = repelem(1:CG.cells.num, diff(CG.cells.facePos))';
allCellFaces = CG.cells.faces(:, 1);
allFaceNormals = CG.faces.normals(allCellFaces, :);

% Nbrs
nbrPairs = CG.faces.neighbors(allCellFaces, :);
areNbrs = bsxfun(@ne, nbrPairs, allCellIndices);
rsNbrPairs = reshape(nbrPairs', [size(nbrPairs, 1) * size(nbrPairs, 2), 1]) ;
rsAreNbrs = reshape(areNbrs', [size(areNbrs, 1) * size(areNbrs, 2), 1]);
allNbrs = rsNbrPairs(rsAreNbrs);

% Get sign if flip has been made. Flip face normals.
nbrs = ~ismember(CG.faces.neighbors(allCellFaces, :), allCellIndices);
nbrs = nbrs(:, 1);
signs = 1 - 2 * nbrs;
allFaceNormals(signs == -1, :) = -allFaceNormals(signs == -1, :);

end

