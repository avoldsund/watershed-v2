%% exampleTOF
G = cartGrid([3, 3], [30, 30]);
G = computeGeometry(G);
heights = [20, 15, 10; 15, 10, 5; 10, 5, 0;];

flux = zeros(G.faces.num, 1);
edgesOne = [16, 17, 18];
edgesTwo = [19, 20, 21];
edgesLeft = [2, 6, 10];
edgesRight = [3, 7, 11];
boundaryLeft = [9, 5, 1];
boundaryRight = [4, 8, 12];

% Scenario 1 - Downwards plane:
% flux(edgesOne) = -7.5;
% flux(edgesTwo) = -12.5;

% Scenario 2 - Plane to the left:
% flux(edgesLeft) = -7.5;
% flux(edgesRight) = -12.5;

% Scenario 3 - Plane towards right corner:
flux(edgesLeft) = 5;
flux(edgesRight) = 5;
flux(edgesOne) = -5;
flux(edgesTwo) = -5;
flux(boundaryRight) = 5;

% State, rock and source
state = struct('flux', flux);
rock = struct('poro', ones(G.cells.num, 1));
srcCell = 3;
src = addSource([], srcCell, -1);

t = computeTimeOfFlight(state, G, rock, 'src', src, 'maxTOF', 1e4, 'reverse', true);
t = t - min(t);
clf,plotCellData(G,t)