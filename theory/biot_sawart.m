clc; clear; close all;

% Define wire segment (current in the +z direction)
N = 20; % Number of wire segments
x_wire = linspace(-1, 1, N);
y_wire = zeros(size(x_wire));
z_wire = zeros(size(x_wire));
I = 1; % Current magnitude (arbitrary units)

% Define observation grid
[X, Y] = meshgrid(linspace(-1.5, 1.5, 20), linspace(-1.5, 1.5, 20));
Z = zeros(size(X));

% Initialize field components
Bx = zeros(size(X));
By = zeros(size(X));
Bz = zeros(size(X));

% Biot-Savart Law Calculation
mu0 = 4*pi*1e-7; % Permeability of free space
dl = [diff(x_wire); diff(y_wire); diff(z_wire)]; % Wire segments

for i = 1:N-1
    % Midpoint of wire segment
    xm = (x_wire(i) + x_wire(i+1)) / 2;
    ym = (y_wire(i) + y_wire(i+1)) / 2;
    zm = (z_wire(i) + z_wire(i+1)) / 2;
    
    % Vector from wire segment to observation points
    Rx = X - xm;
    Ry = Y - ym;
    Rz = Z - zm;
    R = sqrt(Rx.^2 + Ry.^2 + Rz.^2);
    
    % Apply Biot-Savart Law
    dL = [dl(1, i); dl(2, i); dl(3, i)];
    R_vec = [Rx(:), Ry(:), Rz(:)]'; % Convert to 3xN matrix
    dB = (mu0 * I / (4*pi)) * cross(repmat(dL, 1, numel(Rx)), R_vec) ./ (R(:)'.^3);
    
    % Sum contributions
    Bx = Bx + reshape(dB(1, :), size(X));
    By = By + reshape(dB(2, :), size(Y));
    Bz = Bz + reshape(dB(3, :), size(Z));
end

% Normalize for visualization
B_magnitude = sqrt(Bx.^2 + By.^2);
Bx = Bx ./ B_magnitude;
By = By ./ B_magnitude;

% Plot wire
figure; hold on;
plot3(x_wire, y_wire, z_wire, 'k', 'LineWidth', 2);
scatter3(x_wire, y_wire, z_wire, 'ro', 'filled');

% Plot vector field
quiver(X, Y, Bx, By, 'b');

% Labels
xlabel('x'); ylabel('y'); zlabel('z');
title('Magnetic Field from Biot-Savart Law');
axis equal; grid on;
view(2); % 2D top-down view
