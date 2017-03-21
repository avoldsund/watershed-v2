function flux = calculateFlux(CG, faceNormals, faceFlowDirections)
%CALCULATEFLUX Something
%   
    flux = sum(faceNormals .* faceFlowDirections, 2);
    
    N = size(CG.cells.faces, 1);
    deltaZ = zeros(N, 1);
    
    for i = 1:CG.cells.num
        indices = CG.cells.facePos(i) : CG.cells.facePos(i + 1) - 1;
        [faceFlipped, nrmls, sign] = util.flipNormalsOutwards(CG, i);
        faceFluxes = sign .* flux(indices);
        
        nbrs = CG.faces.neighbors(faceFlipped, :);
        nbrs(sign == -1, :) = fliplr(nbrs(sign == -1, :));
        invalidIndices = nbrs(:, 1) == 0 | nbrs(:, 2) == 0;
        validIndices = nbrs(:, 1) ~= 0 & nbrs(:, 2) ~= 0;

        ixValid = indices(validIndices);
        ixInvalid = indices(invalidIndices);
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
    
    flux = flux .* deltaZ;
    flux = flux * 0.1;
end