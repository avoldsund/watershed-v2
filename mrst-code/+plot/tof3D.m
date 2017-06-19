function [] = tof3D(G)
    %TOF3D Summary of this function goes here
    %   Detailed explanation goes here
    
    vertices = sortNodes(G);
    heights = G.cells.z(G.cells.indexMap);
    repHeights = repelem(heights, 4)';
    
    patch('vertices', [G.nodes.coords(vertices), repHeights], 'faces', ', 'EdgeColor', 'None', 'FaceColor', 'blue')
    
end

