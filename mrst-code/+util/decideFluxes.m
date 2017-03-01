function flux = decideFluxes(CG, faceIndices, dotProduct, average)
%DECIDEFLUXES Summary of this function goes here
%   Detailed explanation goes here

% Do the average flux
if average
    a = horzcat(faceIndices, dotProduct);
    b = sortrows(a, 1);  % Sort rows based on face indices
    flux = accumarray(b(:,1), b(:,2)) ./ accumarray(b(:,1), 1);
% Do upflow flux
else
    N = size(faceIndices, 1);
    cellIndices = zeros(N, 1);
    for i = 1:CG.cells.num
       ix = CG.cells.facePos(i):CG.cells.facePos(i+1)-1;
       cellIndices(ix) = i; 
    end
    a = horzcat(cellIndices, faceIndices, dotProduct);
    a = sortrows(a, 2);
    
end


end

