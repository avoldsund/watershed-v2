function rock = setPorosity(CG, nrOfTraps, trapScale)
%SETPOROSITY Set porosity for the cells.
%   ROCK = SETPOROSITY(CG, NROFTRAPS, SCALETRAPPOROSITY) sets the porosity 
%   for every cell in the coarse grid. The cells that are trap cells will
%   be scaled with TRAPSCALE so they take shorter time to fill, simulating
%   that they are already full.

rock = struct('poro', ones(CG.cells.num, 1));
n = CG.cells.num - nrOfTraps + 1;
oneVec = ones(nrOfTraps, 1);
rock.poro(n:end) = oneVec * trapScale;

end

