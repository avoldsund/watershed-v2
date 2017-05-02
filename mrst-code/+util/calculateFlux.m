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
    
    for i = 1:CG.cells.num
        % Identify which faces have outflow and inflow
        indices = CG.cells.facePos(i) : CG.cells.facePos(i + 1) - 1;
        [faceFlipped, ~, sign] = util.flipNormalsOutwards(CG, i);
        faceFluxes = sign .* flux(indices);
        
        nbrs = CG.faces.neighbors(faceFlipped, :);
        nbrs(sign == -1, :) = fliplr(nbrs(sign == -1, :));
        validIndices = nbrs(:, 1) ~= 0 & nbrs(:, 2) ~= 0;

        ixValid = indices(validIndices);
        validNbrs = nbrs(validIndices, :);
        validD = faceFluxes(validIndices, :);
        outFlow = validD > 0;
        inFlow = validD < 0;
        
        deltaOutFlow = CG.cells.z(validNbrs(outFlow, 1)) - CG.cells.z(validNbrs(outFlow, 2));
        deltaInFlow = CG.cells.z(validNbrs(inFlow, 2)) - CG.cells.z(validNbrs(inFlow, 1));
        
        ixOut = ixValid(outFlow);
        posOut = deltaOutFlow > 0;
        negOut = deltaOutFlow <= 0;
        deltaZ(ixOut(negOut)) = 0.1;
        deltaZ(ixOut(posOut)) = deltaOutFlow(posOut);
        
        ixIn = ixValid(inFlow);
        posIn = deltaInFlow > 0;
        negIn = deltaInFlow <= 0;        
        deltaZ(ixIn(negIn)) = 0.1;
        deltaZ(ixIn(posIn)) = deltaInFlow(posIn);
    end
    
    alpha = deltaZ ./ deltaX;
    flux = flux .* alpha;
end
