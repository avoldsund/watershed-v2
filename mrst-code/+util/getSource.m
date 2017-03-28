function [src, trapNr] = getSource(CG, outletCoord, traps, nCols, srcStrength)
%GETSOURCE Summary of this function goes here
%   Detailed explanation goes here

nrOfTraps = size(traps, 1);
outlet = double(outletCoord);
%outlet = [10 * outlet(2), 10 * nCols - 10 * outlet(1)];

% Check if outlet is in a trap
for i = 1:nrOfTraps
    trapCoords = double(horzcat(traps{i,1}', traps{i,2}'));
    
    if any(trapCoords(:, 1) == outlet(1) & trapCoords(:, 2) == outlet(2)) == 1
        srcIx = CG.cells.num - nrOfTraps + i;
        src = addSource([], srcIx, -srcStrength);
        trapNr = i;
        return;
    end
end

noSrc = true;
outlet = [10 * outlet(2), 10 * nCols - 10 * outlet(1)];
distance = util.calculateEuclideanDist(CG.parent.cells.centroids, outlet);
ix = 1:CG.parent.cells.num;
minIx = horzcat(ix', distance);
minIx = sortrows(minIx, 2);
count = 1;
while noSrc
    potSrc = minIx(count, 1);
    potSrc = CG.partition(potSrc);
    if sum(sum(CG.faces.neighbors(util.getCellFaces(CG, potSrc), :))) == 4 * potSrc
       count = count + 1;
    else
        noSrc = false;
        src = potSrc;
    end
end

%src = src + 1; % ONLY FOR THE LARGE WATERSHED
src = addSource([], src, -srcStrength);
trapNr = NaN;

end

