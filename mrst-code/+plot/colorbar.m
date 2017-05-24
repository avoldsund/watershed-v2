function f = colorbar(tof)
    %COLORBAR Summary of this function goes here
    %   Detailed explanation goes here
    
% Create custom color mapping
scale = 255;
greenDark = [0, 122, 0] ./ scale;
greenLight = [102, 194, 165] ./ scale;
colors = [greenLight; greenDark];
x = [0, scale];
map = interp1(x/scale, colors, linspace(0, 1, scale));

% Create figure, set colormap, plot grid and face colors
f = figure('position', [100, 100, 1000, 1000]);
colormap('jet')

% Add frame/invisible line
%color = [0., 0., 0.];
%plot([40, 40, 50, 50, 40], [5, 55, 55, 5, 5], 'Color', color)
%axis off;
c = colorbar;
set(c, 'fontsize', 30);
caxis([0, max(tof)]);
axis off;
    
end