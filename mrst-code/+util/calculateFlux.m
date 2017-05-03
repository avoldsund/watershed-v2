function flux = calculateFlux(CG, faceNormals, faceFlowDirections, scale)
%CALCULATEFLUX returns the fluxes given face normals and the flow
%directions over the face.
%   FLUX = CALCULATEFLUX(CG, FACENORMALS, FACEFLOWDIRECTIONS, SCALE) takes
%   the coarse grid CG, altered face normals FACENORMALS and altered face flow
%   directions FACEFLOWDIRECTIONS. If SCALE is true, the fluxes use
%   elevation data to scale them.

    deltaX = CG.faceLength;
    flux = sum(faceNormals .* faceFlowDirections, 2);
    
    if ~scale    
        flux = flux ./ deltaX;
        return
    end
    
    N = size(CG.cells.faces, 1);
    deltaZ = zeros(N, 1);
    
    [~, ~, ~, nbrs, nbrPairs, signs] = util.flipAllNormals(CG);
    allFaceFluxes = signs .* flux;
    
    % Separate faces into outflow and inflow faces
    indices = (1:N)';
    validIndices = nbrs ~= 0;
    ixValid = indices(validIndices);
    validNbrs = nbrPairs(ixValid, :);
    validDp = allFaceFluxes(ixValid, :);
    outFlow = validDp > 0;
    inFlow = validDp < 0;
    
    % Elevation difference
    deltaOutFlow = CG.cells.z(validNbrs(outFlow, 1)) - CG.cells.z(validNbrs(outFlow, 2));
    deltaInFlow = CG.cells.z(validNbrs(inFlow, 2)) - CG.cells.z(validNbrs(inFlow, 1));

    % If elevation differences are positive, use those for alpha, if not,
    % set the scaling to a constant
    % Treat outflow faces
    ixOut = ixValid(outFlow);
    posOut = deltaOutFlow > 0;
    negOut = deltaOutFlow <= 0;
    deltaZ(ixOut(negOut)) = 0.1;
    deltaZ(ixOut(posOut)) = deltaOutFlow(posOut);

    % Treat inflow faces
    ixIn = ixValid(inFlow);
    posIn = deltaInFlow > 0;
    negIn = deltaInFlow <= 0;        
    deltaZ(ixIn(negIn)) = 0.1;
    deltaZ(ixIn(posIn)) = deltaInFlow(posIn);
    
    % Perform derivative/slope scaling
    alpha = deltaZ ./ deltaX;
    flux = flux .* alpha;
    
end
