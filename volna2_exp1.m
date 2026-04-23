clear; clc; close all;

% ПАРАМЕТРЫ
xf = -10; yf = -4; 
x0 = 0; y0 = 0; thetas = pi; 
xi = [x0; y0; thetas];

% Параметры сетки
grid_res = 0.3; 
x_vec = -15:grid_res:2;
y_vec = -6:grid_res:6;
[XG, YG] = meshgrid(x_vec, y_vec);

% стены
obs_mask = (XG >= -5 & XG <= -3 & YG >= -2 & YG <= 6) | ... 
           (XG >= -9 & XG <= -7 & YG >= -6 & YG <= 2);

% зона эхо
margin = 0.6; % Ширина зоны безопасности
obs_mask_inflated = (XG >= -5-margin & XG <= -3+margin & YG >= -2-margin & YG <= 6+margin) | ...
                    (XG >= -9-margin & XG <= -7+margin & YG >= -6-margin & YG <= 2+margin);

% РАСЧЕТ ТЕПЛОВОЙ КАРТЫ
Total_Heat_Map = Inf(size(XG));
[~, iy_f] = min(abs(y_vec - yf));
[~, ix_f] = min(abs(x_vec - xf));
Total_Heat_Map(iy_f, ix_f) = 0;

queue = [iy_f, ix_f];
dirs = [1,0; -1,0; 0,1; 0,-1; 1,1; 1,-1; -1,1; -1,-1];
d_costs = [1, 1, 1, 1, sqrt(2), sqrt(2), sqrt(2), sqrt(2)] * grid_res;

fprintf('Расчет тепловой карты...\n');
while ~isempty(queue)
    curr = queue(1,:); queue(1,:) = [];
    for i = 1:8
        ny = curr(1) + dirs(i,1); nx = curr(2) + dirs(i,2);
        if ny > 0 && ny <= size(XG,1) && nx > 0 && nx <= size(XG,2)
            if ~obs_mask(ny,nx)
                new_dist = Total_Heat_Map(curr(1), curr(2)) + d_costs(i);
                if new_dist < Total_Heat_Map(ny,nx)
                    Total_Heat_Map(ny,nx) = new_dist;
                    queue = [queue; ny, nx];
                end
            end
        end
    end
end

T = 80; dt = 0.01; Nsteps = floor(T/dt);
W = [0.5; 0.5; 0.5]; 
a1_learn = 0.1; 
R = eye(2); R_inv = inv(R);

traj = zeros(3, Nsteps);

fprintf('Старт...\n');
for k = 1:Nsteps
    traj(:,k) = xi;
    
    [~, cur_ix] = min(abs(x_vec - xi(1)));
    [~, cur_iy] = min(abs(y_vec - xi(2)));
    
    look_ahead = 3; 
    best_v = Total_Heat_Map(cur_iy, cur_ix);
    target_l = [xf, yf]; 
    found = false;
    
    for dy = -look_ahead:look_ahead
        for dx = -look_ahead:look_ahead
            ny = cur_iy + dy; nx = cur_ix + dx;
            if ny > 0 && ny <= size(XG,1) && nx > 0 && nx <= size(XG,2)
                if Total_Heat_Map(ny,nx) < best_v && ~obs_mask_inflated(ny,nx)
                    best_v = Total_Heat_Map(ny,nx);
                    target_l = [XG(ny,nx), YG(ny,nx)];
                    found = true;
                end
            end
        end
    end
    
    % Если в зоне видимости нет точек вне эха (застряли), тянем к финишу
    if ~found, target_l = [xf, yf]; end

    dx = target_l(1) - xi(1);
    dy = target_l(2) - xi(2);
    rho = sqrt(dx^2 + dy^2);
    alpha = wrapToPi(atan2(dy, dx) - xi(3));
    
    rho_near = max(rho, 0.1);

    [phi_vec, dphi_de] = get_basis_reduced(rho, alpha);
    g_polar = [-cos(alpha), 0; sin(alpha)/rho_near, -1];
    
    u = -0.5 * R_inv * g_polar' * (dphi_de' * W);
    
    % Ограничения скоростей
    % u(1) = max(min(u(1), 1.0), -1.0);
    % u(2) = max(min(u(2), 2.5), -2.5);
    
    % ОБНОВЛЕНИЕ ВЕСОВ
    sigma = dphi_de * (g_polar * u);
    Q_cost = rho^2 + alpha^2;
    residual = sigma' * W + Q_cost + u'*R*u;
    W = W - a1_learn * (sigma / (sigma' * sigma + 1)) * residual * dt;

    xi(1) = xi(1) + dt * u(1) * cos(xi(3));
    xi(2) = xi(2) + dt * u(1) * sin(xi(3));
    xi(3) = wrapToPi(xi(3) + dt * u(2));
    
    if sqrt((xi(1)-xf)^2 + (xi(2)-yf)^2) < 0.4, break; end
end

% ГРАФИКИ
figure('Name','Trajectory with Echo Zone','Position', [100 100 850 600]);

% Тепловая карта
contourf(XG, YG, Total_Heat_Map, 30, 'LineColor', 'none', 'HandleVisibility', 'off'); 
hold on; colormap(flipud(jet)); colorbar;

% Траектория
plot(traj(1,1:k), traj(2,1:k), 'b-', 'LineWidth', 2.5, 'DisplayName', 'Trajectory'); 

% Реальные препятствия 
fill([-5 -3 -3 -5], [-2 -2 6 6], [0.4 0.4 0.4], 'DisplayName', 'Wall');
fill([-9 -7 -7 -9], [-6 -6 2 2], [0.4 0.4 0.4], 'HandleVisibility', 'off');

% Зона эха
rect1_x = [-5-margin, -3+margin, -3+margin, -5-margin, -5-margin]; 
rect1_y = [-2-margin, -2-margin, 6+margin, 6+margin, -2-margin];
rect2_x = [-9-margin, -7+margin, -7+margin, -9-margin, -9-margin]; 
rect2_y = [-6-margin, -6-margin, 2+margin, 2+margin, -6-margin];

plot(rect1_x, rect1_y, '--', 'Color', [1 0.7 0.7], 'LineWidth', 1.5, 'HandleVisibility','off');
plot(rect2_x, rect2_y, '--', 'Color', [1 0.7 0.7], 'LineWidth', 1.5, 'HandleVisibility','off');
plot(NaN, NaN, '--', 'Color', [1 0.7 0.7], 'LineWidth', 1.5, 'DisplayName', 'Echo Zone');

% Старт и Цель
plot(x0, y0, 'ko', 'MarkerFaceColor','g', 'MarkerSize',8, 'DisplayName','Start');
plot(xf, yf, 'rx', 'MarkerSize',12, 'LineWidth',3, 'DisplayName','Goal');

xlabel('x [m]'); ylabel('y [m]'); title('Robot trajectory');
axis equal; grid on; 
legend('Location', 'northeastoutside'); 

function [phi, dphide] = get_basis_reduced(rho, alpha)    
    phi = [rho^2; alpha^2; rho*alpha];
    dphide = [2*rho, 0; 0, 2*alpha; alpha, rho];
end

function out = wrapToPi(angle)
    out = mod(angle + pi, 2*pi) - pi;
end