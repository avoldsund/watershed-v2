classdef TestUtil < matlab.unittest.TestCase
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods (Test)
        
        % MapCoordsToIndices
        function testMapCoordsToIndicesSmallQuadratic(testCase)
            coords = [0, 1; 1, 0; 2, 2;]; nCols = 3; nRows = 3;
            actSolution = util.mapCoordsToIndices(coords, nCols, nRows);
            expSolution = [8, 4, 3]';
            testCase.assertEqual(actSolution, expSolution)
        end
        
        function testMapCoordsToIndicesRectangular(testCase)
            coords = [0, 0; 1, 0; 1, 2;]; nCols = 3; nRows = 2; 
            actSolution = util.mapCoordsToIndices(coords, nCols, nRows);
            expSolution = [4, 1, 3]';
            testCase.assertEqual(actSolution, expSolution)
        end
        
        function testMapCoordsToIndicesSingleCoord(testCase)
            coords = [1, 1]; nCols = 3; nRows = 3;
            actSolution = util.mapCoordsToIndices(coords, nCols, nRows);
            expSolution = 5;
            testCase.assertEqual(actSolution, expSolution)
        end

        % MapListOfCoordsToIndices
        function testMapListOfCoordsToIndices(testCase)
            listOfCoords = cell(2,2);
            listOfCoords{1,1} = [1, 2];
            listOfCoords{1,2} = [4, 4];
            listOfCoords{2,1} = [4, 4, 4];
            listOfCoords{2,2} = [1, 2, 3];
            
            nCols = 6; nRows = 6;
            
            actSolution = util.mapListOfCoordsToIndices(listOfCoords, nCols, nRows);
            expSolution = cell(2,1);
            expSolution{1,1} = [29, 23]';
            expSolution{2,1} = [8, 9, 10]';
            testCase.assertEqual(actSolution, expSolution) 
        end
        
        % FixPartitioning
        function testFixPartitioning(testCase)
            nRows = 6; nCols = 6;
            wsIndices = [28, 29, 21, 22, 23, 14, 15, 16, 17, 8, 9, 10, 11];
            
            traps = cell(2,2);
            traps{1,1} = [1, 2];
            traps{1,2} = [4, 4];
            traps{2,1} = [4, 4, 4];
            traps{2,2} = [1, 2, 3];
            nrOfTraps = 2;
            
            G = cartGrid([nCols, nRows]);
            G = computeGeometry(G);
            rmCells = setdiff(1 : nCols * nRows, wsIndices);
            G = removeCells(G, rmCells);
            
            actSolution = util.fixPartitioning(G, traps, nrOfTraps);
            expSolution = [10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 9, 8, 9]';
            testCase.assertEqual(actSolution, expSolution)
        end
        
        function testFixPartitioningThreeTraps(testCase)
            nRows = 6; nCols = 6;
            wsIndices = [28, 29, 21, 22, 23, 14, 15, 16, 17, 8, 9, 10, 11];
            
            traps = cell(3,2);
            traps{1,1} = [1, 2];
            traps{1,2} = [4, 4];
            traps{2,1} = [4, 4, 4];
            traps{2,2} = [1, 2, 3];
            traps{3,1} = 2;
            traps{3,2} = 2;
            nrOfTraps = 3;
            
            G = cartGrid([nCols, nRows]);
            G = computeGeometry(G);
            rmCells = setdiff(1 : nCols * nRows, wsIndices);
            G = removeCells(G, rmCells);
            
            actSolution = util.fixPartitioning(G, traps, nrOfTraps);
            expSolution = [9, 9, 9, 1, 2, 3, 4, 5, 10, 6, 8, 7, 8]';
            testCase.assertEqual(actSolution, expSolution)
        end
        
        % SetHeightsCoarseGrid
        function testSetHeightsCoarseGrid(testCase)
            nRows = 6; nCols = 6;
            wsIndices = [28, 29, 21, 22, 23, 14, 15, 16, 17, 8, 9, 10, 11];
            heights = [10, 10, 10, 10, 10, 10; 10, 9, 9, 9, 7, 10;
                       10, 9, 10, 9, 7, 10; 8, 10, 10, 10, 7, 10;
                       10, 4, 4, 4, 4.5, 10; 10, 4, 10, 10, 10, 10;];
            heights = rot90(heights, -1);
            
            G = cartGrid([nCols, nRows]);
            G = computeGeometry(G);
            rmCells = setdiff(1 : nCols * nRows, wsIndices);
            G = removeCells(G, rmCells);
            
            partition = [10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 9, 8, 9]';

            CG = generateCoarseGrid(G, partition);
            CG = coarsenGeometry(CG);
            
            trapHeights = [7, 4]';
            nrOfTraps = 2;
            
            actSolution = util.setHeightsCoarseGrid(CG, heights, trapHeights, nrOfTraps);
            expSolution = [4.5, 10, 10, 10, 7, 10, 9, 9, 7, 4]';
            testCase.assertEqual(actSolution, expSolution)
        end
        
        function testSetHeightsCoarseGridThreeTraps(testCase)
            nRows = 6; nCols = 6;
            wsIndices = [28, 29, 21, 22, 23, 14, 15, 16, 17, 8, 9, 10, 11];
            heights = [10, 10, 10, 10, 10, 10; 10, 9, 9, 9, 7, 10;
                       10, 9, 10, 9, 7, 10; 8, 10, 10, 10, 7, 10;
                       10, 4, 4, 4, 4.5, 10; 10, 4, 10, 10, 10, 10;];
            heights = rot90(heights, -1);
            
            G = cartGrid([nCols, nRows]);
            G = computeGeometry(G);
            rmCells = setdiff(1 : nCols * nRows, wsIndices);
            G = removeCells(G, rmCells);
            
            partition = [9, 9, 9, 1, 2, 3, 4, 5, 10, 6, 8, 7, 8]';

            CG = generateCoarseGrid(G, partition);
            CG = coarsenGeometry(CG);
            
            trapHeights = [7, 4, 10]';
            nrOfTraps = 3;
            
            actSolution = util.setHeightsCoarseGrid(CG, heights, trapHeights, nrOfTraps);
            expSolution = [4.5, 10, 10, 10, 7, 9, 9, 7, 4, 10]';
            testCase.assertEqual(actSolution, expSolution)
        end
        
        % GetFlowDirections
        function testGetFlowDirections(testCase)
            watershed = [28, 29, 21, 22, 23, 14, 15, 16, 17, 8, 9, 10, 11];
            heights = [10, 10, 10, 10, 10, 10; 10, 9, 9, 9, 7, 10;
                       10, 9, 10, 9, 7, 10; 8, 10, 10, 10, 7, 10;
                       10, 4, 4, 4, 4.5, 10; 10, 4, 10, 10, 10, 10;];
            heights = rot90(heights, -1);  % Fix 1d-indexing
            traps = cell(2,2);
            traps{1,1} = [1, 2];
            traps{1,2} = [4, 4];
            traps{2,1} = [4, 4, 4];
            traps{2,2} = [1, 2, 3];
            nrOfTraps = 2;
            
            spillPairs = [23, 9]';
            flowDirections = [0, 0, 0, 0, 0, 0; 
                              0, -1, -1, 2, -1, 0;
                              0, 16, 2, 2, 8, 0;
                              0, 8, 8, 8, 8, 0;
                              0, -1, 16, -1, 32, 0;
                              0, 0, 0, 0, 0, 0];
            fd = rot90(flowDirections, -1);  % Fix 1d-indexing
            
            CG = util.createCoarseGrid(watershed, heights, traps, nrOfTraps);
            CG.cells.fd = [32, 8, 8, 8, 8, 2, 2, 2, 8, 16]';
            
            actSolution = util.getFlowDirections(CG, fd, nrOfTraps, spillPairs);
            expSolution = [-1, 0; 0, -1; 0, -1; 0, -1; 0, -1; 1, 0; 1, 0; 1, 0; 0, -1; -1, -1;];
            testCase.assertEqual(actSolution, expSolution)
        end
        
        
    end
end
