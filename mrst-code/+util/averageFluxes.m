function flux = averageFluxes(faceIndices, flux)
%AVERAGEFLUXES Take average of fluxes for faces with multiple defined
%fluxes.
%   FLUX = AVERAGEFLUXES(FACEINDICES, FLUX) returns the average FLUX for
%   faces with multiple defined fluxes. Faces which are not multiply
%   defined are not affected.

a = horzcat(faceIndices, flux);
b = sortrows(a, 1);  % Sort rows based on face indices
flux = accumarray(b(:,1), b(:,2)) ./ accumarray(b(:,1), 1);

end