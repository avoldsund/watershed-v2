%% Calculate hydrograph for a precipitation scenario in a DEM landscape

% If it doesn't work properly, change src in util.getSource

phi = 10^-1;
scaleFluxes = true;
[CG, tof] = calculateTof(phi, scaleFluxes);
tof = ceil(tof);

%% Make moving front

% Define movement
d = [0, 1]; % No need to normalize
frontSize = 25;
offset = frontSize / 2;

minCoord = min(CG.faces.centroids);
minX = minCoord(1);
minY = minCoord(2);
maxCoord = max(CG.faces.centroids);
maxX = maxCoord(1);
maxY = maxCoord(2);

if d(1) ~= 0 % Move horizontally
    w = frontSize;
    l = maxY - minY;
    originX = minX - w;
    originY = minY;
    cornersY = [originY, originY + l, originY + l, originY];
    if d(1) > 0 % Move east
        cornersX = [originX, originX, originX + w, originX + w] + offset;
        center = [originX + offset + w/2, originY + l/2];
    else % Move west
        offset = (maxX - minX) + w - offset;
        cornersX = [originX, orig10inX, originX + w, originX + w] + offset;
        center = [originX + offset + w/2, originY + l/2];
    end
    
else
    l = maxX - minX;
    w = frontSize;
    originX = minX;
    originY = minY - w;
    cornersX = [originX, originX, originX + l, originX + l];
    if d(2) > 0 % Move north
        cornersY = [originY, originY + w, originY + w, originY] + offset;
        center = [originX + l/2, originY + w/2 + offset];
    else % Move south
        offset = (maxY - minY) + w - offset;
        cornersY = [originY, originY + w, originY + w, originY] + offset;
        center = [originX + l/2, originY + w/2 + offset];
    end
end
corners = [cornersX; cornersY]';

intensity = 10; % mm/hour
v = 1; % m/s
gaussian = true;
maxTime = 10^3;

front = struct('amplitude', intensity,...
               'velocity', v,...
               'direction', d,...
               'frontSize', frontSize,...
               'center', center,...
               'corners', corners,...
               'gaussian', gaussian);

timeSteps = [1, 2, 4, 8, 16];
discharges = zeros(size(timeSteps, 2), maxTime);
for i = 1:size(timeSteps, 2)
    timeStep = timeSteps(i);
    front.center = [front.center(1) + front.velocity * front.direction(1, 1) * timeStep/2, ...
        front.center(2) + front.velocity * front.direction(1, 2) * timeStep/2];
    front.corners = bsxfun(@plus, front.corners, front.velocity .* front.direction * timeStep/2);
    discharges(i, :) = util.hydrographMovingFrontFast(CG, tof, front, timeStep, maxTime);
end

%%
errors_l2 = zeros(1, size(timeSteps, 2));
errors_max = zeros(1, size(timeSteps, 2));
max_discharge = zeros(1, size(timeSteps, 2));
dischargeRef = discharges(1, :)';
figure; hold on;
plot(dischargeRef)

for i = 2:size(timeSteps, 2)
    i
    timeStep = timeSteps(i);
    discharge = discharges(i, :);
    b = arrayfun(@(j) mean(discharge(j:j+timeStep-1)),1:timeStep:length(discharge)-timeStep+1)'; % the averaged vector
    c = (timeStep/2):timeStep:numel(discharge);
    if size(c, 2) > size(b, 1)
        c = c(1:end-1);
    end
    plot(c, b,'LineWidth', 2)
    errors_l2(i) = sqrt(timeStep) * sqrt(sum(abs(dischargeRef(c) - b).^2));
    errors_max(i) = max(abs(dischargeRef(c) - b));
    
    errors_l2(i) = sqrt(timeStep) * sqrt(sum(abs(dischargeRef(c)- b).^2)) ...
        / norm(dischargeRef);
    errors_max(i) = max(abs(dischargeRef(c) - b));
    max_discharge(i) = abs(max(dischargeRef(c) - max(b)));
end

xlabel('Time (s)')
ylabel('Discharge{(}m^3/s)')

legend('1', '2', '4', '8', '16', '21')

%% Plot error

figure;loglog([1, 2, 4, 8, 16], errors_l2, '*-', 'LineWidth', 3)
hold on;loglog([1, 2, 4, 8, 16], [1, 2, 4, 8, 16]*10^-1, 'LineWidth', 3)
set(gca, 'FontSize', 24);
xlab = xlabel('$\Delta t(s)$'); set(xlab,'Interpreter','latex');
ylab = ylabel('$\mathcal{E}_2$'); set(ylab,'Interpreter','latex');


