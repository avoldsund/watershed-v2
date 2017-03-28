function [nbrs] = investigateMaxTof(CG, tof, maxTime, faceFlowDirections)

undefined = find(tof == maxTime)

for i = 1:size(undefined, 1)
    c = undefined(i);
    startIx = CG.cells.facePos(c)
    endIx = CG.cells.facePos(c + 1) - 1;
    interval = startIx:endIx;
    [faceFlipped, faceNormals, sign] = util.flipNormalsOutwards(CG, c);
    d = sum(faceNormals .* faceFlowDirections(startIx:endIx, :), 2);
    nbrs = CG.faces.neighbors(faceFlipped, :);
    nbrs(sign == -1, :) = fliplr(nbrs(sign == -1, :));
    nbrs = nbrs(d > 0, :)
    if unique(nbrs(nbrs ~= 0))
        continue
    end
    
end
nbrs = NaN;
end

