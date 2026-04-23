clear; clc; close all;

% ПАРАМЕТРЫ
xf = -11; yf = 0; 
x0 = 0; y0 = 0; thetas = pi; 
xi = [x0; y0; thetas];

% Параметры сетки
grid_res = 0.2; % разрешение сетки
x_vec = -15:grid_res:2;
y_vec = -6:grid_res:6;
[XG, YG] = meshgrid(x_vec, y_vec);

% стена
obs_mask = (XG >= -6 & XG <= -4 & YG >= -4 & YG <= 4);

% РАСЧЕТ
Total_Heat_Map = Inf(size(XG));
[~, iy_f] = min(abs(y_vec - yf));
[~, ix_f] = min(abs(x_vec - xf));
Total_Heat_Map(iy_f, ix_f) = 0;

queue = [iy_f, ix_f];
dirs = [1,0; -1,0; 0,1; 0,-1; 1,1; 1,-1; -1,1; -1,-1];
d_costs = [1, 1, 1, 1, sqrt(2), sqrt(2), sqrt(2), sqrt(2)] * grid_res;

disp('Расчет волнового фронта...');
while ~isempty(queue)
    curr = queue(1,:);
    queue(1,:) = [];
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
disp('Волновой фронт готов.');

%ПАРАМЕТРЫ
T = 60;        
dt = 0.01;      
Nsteps = floor(T/dt);

N_basis = 3; 
W = zeros(N_basis,1) + 0.5; 
a1 = 0.002;                   
R = [1 0; 0 1];
R_inv = inv(R);

traj = zeros(3, Nsteps);
u_log = zeros(2, Nsteps);
W_history = zeros(N_basis, Nsteps);
t_log = (0:Nsteps-1)*dt;

%% 4. ОСНОВНОЙ ЦИКЛ
for k = 1:Nsteps
    traj(:,k) = xi;
    
    % Поиск локальной цели по градиенту волнового фронта
    [~, cur_ix] = min(abs(x_vec - xi(1)));
    [~, cur_iy] = min(abs(y_vec - xi(2)));
    
    look_ahead = 5 ; % смотрит в радиусе 4 лучшую точку туда и едет
    best_val = Total_Heat_Map(cur_iy, cur_ix);
    target_l = [xf, yf];
    
    for dy = -look_ahead:look_ahead
        for dx = -look_ahead:look_ahead
            ny = cur_iy + dy; nx = cur_ix + dx;
            if ny > 0 && ny <= size(XG,1) && nx > 0 && nx <= size(XG,2)
                if Total_Heat_Map(ny,nx) < best_val
                    best_val = Total_Heat_Map(ny,nx);
                    target_l = [XG(ny,nx), YG(ny,nx)]; % временная цель
                end
            end
        end
    end

    % Ошибки относительно локальной цели
    dx = xi(1) - target_l(1);
    dy = xi(2) - target_l(2);
    rho = sqrt(dx^2 + dy^2);
    phi = atan2(dy, dx);
    alpha = wrapToPi(phi - xi(3) + pi);
    
    if rho < 0.01, rho = 0.01; end

    [phi_vec, dphi_de] = get_basis_reduced(rho, alpha);
    gradV_e = dphi_de' * W; 
    g_polar = [-cos(alpha), 0; sin(alpha)/rho, -1];
    u = -0.5 * R_inv * g_polar' * gradV_e;
    
    % Ограничения
    % u(1) = max(min(u(1), 1.0), -1.0);
    % u(2) = max(min(u(2), 1.5), -1.5);
    u_log(:,k) = u;
    
    % Обновление весов
    sigma = dphi_de * (g_polar * u);
    Q_cost = 1*rho^2 + 1*alpha^2; 
    U_cost = u' * R * u;
    residual = sigma' * W + Q_cost + U_cost;
    Wdot = - a1 * (sigma / (sigma' * sigma + 1)) * residual;
    W = W + dt * Wdot;
    W_history(:,k) = W;
    
    % Интегрирование
    xi(1) = xi(1) + dt * u(1) * cos(xi(3));
    xi(2) = xi(2) + dt * u(1) * sin(xi(3));
    xi(3) = wrapToPi(xi(3) + dt * u(2));
    
    if sqrt((xi(1)-xf)^2 + (xi(2)-yf)^2) < 0.2
        break;
    end
end

%ГРАФИКИ

% ТРАЕКТОРИЯ
figure('Name','Trajectory');

% 1. Добавляем 'HandleVisibility', 'off', чтобы скрыть тепловую карту из легенды
contourf(XG, YG, Total_Heat_Map, 30, 'LineColor', 'none', 'HandleVisibility', 'off'); 
hold on; colormap(flipud(jet)); colorbar;

% 2. Добавляем 'DisplayName', 'Trajectory', чтобы вместо "data" было понятное имя
plot(traj(1,1:k), traj(2,1:k), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Trajectory'); 

% Препятствие (уже имеет имя, оставляем)
fill([-6 -4 -4 -6], [-4 -4 4 4], [0.4 0.4 0.4], 'DisplayName', 'Obstacle');

% Пунктир зоны эха
rectangle('Position',[-6.8 -4.8 3.8 9.8], 'Curvature',0.2, 'LineStyle','--', 'EdgeColor', [1 0.7 0.7]);
plot(NaN, NaN, '--', 'Color', [1 0.7 0.7], 'LineWidth', 1.5, 'DisplayName', 'Echo Zone');

% Старт и Цель
plot(x0, y0, 'ko', 'MarkerSize',8, 'DisplayName','Start');
plot(xf, yf, 'rx', 'MarkerSize',12, 'LineWidth',2, 'DisplayName','Goal');

xlabel('x'); ylabel('y'); title('Траектория');
axis equal; grid on; 
legend('Location', 'northeastoutside'); 

  % ФУНКЦИИ
function [phi, dphide] = get_basis_reduced(rho, alpha)    
    phi = [rho^2; alpha^2; rho*alpha];
    dphide = [2*rho, 0; 0, 2*alpha; alpha, rho];
end