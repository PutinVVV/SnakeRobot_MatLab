clear; clc; close all;

% ПАРАМЕТРЫ СИСТЕМЫ
N = 4;             
m = 0.3;       
l_half = 0.1;      
J = (1/3) * m * (2*l_half)^2; 

ct = 0.05; cn = 20.0;          
cp = (cn - ct) / (2 * l_half);

omega_val = 10; delta_opt = 1.0; 
lambda1 = 1.0; lambda2 = 25.0; 

xf = -11; yf = -3; % Цель

% сетка для волнового
grid_res = 0.4; 
x_vec = -20:grid_res:5;
y_vec = -20:grid_res:5;
[XG, YG] = meshgrid(x_vec, y_vec);

% Физическая стена 
obs_mask = (XG > -10 & XG < -8 & YG > -12 & YG < 0) | ...
           (XG > -12 & XG < -10 & YG > -2 & YG < 0);

% Зона эха 
obs_mask_inflated = (XG > -10.8 & XG < -7.2 & YG > -12.8 & YG < 0.8) | ...
                    (XG > -12.8 & XG < -9.2 & YG > -2.8 & YG < 0.8);

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

% МАТРИЦЫ И ADP
A = zeros(N-1, N); D = zeros(N-1, N);
for i = 1:N-1
    D(i, i) = 1; D(i, i+1) = -1;
    A(i, i) = 1; A(i, i+1) = 1;
end
inv_DDT = inv(D * D' + 1e-4*eye(N-1));
K_pos = D' * inv_DDT * A; V_mat = A' * inv_DDT * A; K_mat = A' * inv_DDT * D; 
e_vec = ones(N, 1);
AD_T = A * D';
k_delta = abs(sum(sum(AD_T .* sin(( (1:N-1)' - (1:N-1) ) * delta_opt))));

N_basis = 3; 
W = [0.4; 0.4; 0.4]; % Исходные веса
a1_learn = 0.2;      % Исходная скорость обучения
R_mat = diag([1, 1]); R_inv = inv(R_mat);

% СИМУЛЯЦИЯ
dt = 0.0001; T_end = 60; Nsteps = floor(T_end/dt);
y = zeros(2*N + 4, 1);
y(1:N) = linspace(0.02, -0.02, N); 

traj_log = zeros(2, Nsteps);
theta_log = zeros(N, Nsteps);
W_log = zeros(N_basis, Nsteps);

fprintf('Старт...\n');
for k = 1:Nsteps
    theta = y(1:N); pos = y(N+1:N+2);
    dtheta = y(N+3:2*N+2); dpos = y(2*N+3:2*N+4);
    theta_bar = mean(theta); 
    phi = D * theta; dphi = D * dtheta;
    vt_curr = dpos(1)*cos(theta_bar) + dpos(2)*sin(theta_bar);

    % выбор цели
    [~, cur_ix] = min(abs(x_vec - pos(1)));
    [~, cur_iy] = min(abs(y_vec - pos(2)));
    
    look_ahead = 6; 
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
    
    % Если мы застряли (все соседи хуже или в зоне эха), плавно тянем к глобальной цели
    if ~found, target_l = [xf, yf]; end

    % ОШИБКИ 
    dx = target_l(1) - pos(1); dy = target_l(2) - pos(2);
    rho = sqrt(dx^2 + dy^2);
    alpha_err = wrapToPi(atan2(dy, dx) - theta_bar + pi);
    
    rho_s = rho / 0.8; % Нормализация для обучения

    [~, dphi_de] = get_basis_reduced(rho_s, alpha_err);
    g_polar = [-cos(alpha_err),            0; 
                sin(alpha_err)/(rho_s+0.5), -1];
            
    u_rl = -0.5 * R_inv * g_polar' * (dphi_de' * W);
    v_req = max(abs(u_rl(1)), 0.3); 
    w_req = u_rl(2); 

    % ОБНОВЛЕНИЕ ВЕСОВ
    sigma_adp = dphi_de * (g_polar * u_rl);
    residual = sigma_adp' * W + (rho_s^2 + alpha_err^2);
    W = W - a1_learn * (sigma_adp / (sigma_adp' * sigma_adp + 1)) * residual * dt;

    % Динамика...
    amp = 0.2 + sqrt( (2 * N * ct * v_req) / (cp * omega_val * k_delta + 1e-4) );
    phi_0 = (lambda1 * w_req) / (lambda2 * max(abs(vt_curr), 0.2));
    t_c = k * dt;
    phase = omega_val * t_c + (1:N-1)' * delta_opt;
    phi_ref = amp * sin(phase) + phi_0;
    phid_ref = amp * omega_val * cos(phase);
    phidd_ref = -amp * omega_val^2 * sin(phase);
    u_torques = phidd_ref + 150 * (phi_ref - phi) + 20 * (phid_ref - dphi);
    u_torques = max(min(u_torques, 20), -20); 

    S_th = diag(sin(theta)); C_th = diag(cos(theta));
    M_th = J * eye(N) + m * l_half^2 * (S_th * V_mat * S_th + C_th * V_mat * C_th) + 1e-5*eye(N);
    W_th = m * l_half^2 * (S_th * V_mat * C_th - C_th * V_mat * S_th); 
    vx_i = dpos(1)*e_vec + l_half * K_pos * S_th * dtheta; 
    vy_i = dpos(2)*e_vec - l_half * K_pos * C_th * dtheta;
    frx = -(ct*cos(theta).^2 + cn*sin(theta).^2).*vx_i - (ct-cn)*sin(theta).*cos(theta).*vy_i; 
    fry = -(ct-cn)*sin(theta).*cos(theta).*vx_i - (ct*sin(theta).^2 + cn*cos(theta).^2).*vy_i;
    tau_fric = - l_half * S_th * K_mat * frx + l_half * C_th * K_mat * fry;
    ddtheta = M_th \ (D' * u_torques - W_th * (dtheta.^2) - tau_fric);
    ddpos = (1/(N*m)) * [sum(frx); sum(fry)];
    y = y + [dtheta; dpos; ddtheta; ddpos] * dt;

    traj_log(:,k) = pos;
    theta_log(:,k) = theta;
    W_log(:,k) = W;

    if sqrt((pos(1)-xf)^2 + (pos(2)-yf)^2) < 0.5, break; end
end

traj_log = traj_log(:, 1:k); W_log = W_log(:, 1:k);
theta_log = theta_log(:, 1:k); t_plot = (0:k-1)*dt;

% ГРАФИКИ 
figure('Name','Trajectory','Color','w');

% Рисуем  тепловую карту
contourf(XG, YG, Total_Heat_Map, 30, 'LineColor', 'none', 'HandleVisibility', 'off'); 
hold on; colormap(flipud(jet)); colorbar;

% Рисуем траекторию ЦМ 
z_line = 5 * ones(size(traj_log(1,:)));
plot3(traj_log(1,:), traj_log(2,:), z_line, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Trajectory'); 

%  Препятствие
fill3([-10 -8 -8 -10], [-12 -12 0 0], [2 2 2 2], [0.4 0.4 0.4], 'DisplayName', 'Wall');
fill3([-12 -10 -10 -12], [-2 -2 0 0], [2 2 2 2], [0.4 0.4 0.4], 'HandleVisibility', 'off');

% Зона эха 
rect1_x = [-10.8, -7.2, -7.2, -10.8, -10.8]; rect1_y = [-12.8, -12.8, 0.8, 0.8, -12.8];
rect2_x = [-12.8, -9.2, -9.2, -12.8, -12.8]; rect2_y = [-2.8, -2.8, 0.8, 0.8, -2.8];
plot3(rect1_x, rect1_y, 3*ones(size(rect1_x)), '--', 'Color', [1 0.7 0.7], 'LineWidth', 1.5, 'HandleVisibility','off');
plot3(rect2_x, rect2_y, 3*ones(size(rect2_x)), '--', 'Color', [1 0.7 0.7], 'LineWidth', 1.5, 'HandleVisibility','off');
plot(NaN, NaN, '--', 'Color', [1 0.7 0.7], 'LineWidth', 1.5, 'DisplayName', 'Echo Zone');

%  Точки Старта и Цели 
plot3(0, 0, 10, 'ko', 'MarkerFaceColor','g', 'MarkerSize', 8, 'DisplayName', 'Start');
plot3(xf, yf, 10, 'rx', 'MarkerSize', 12, 'LineWidth', 3, 'DisplayName', 'Goal');

view(2); 
xlabel('x [m]'); ylabel('y [m]'); title('Robot Trajectory on Heat Map');
axis equal; grid on; 
legend('Location','northeastoutside');

% АНИМАЦИЯ 
figure('Name','Snake Animation','Position', [100, 100, 900, 800], 'Color', 'w');
axis equal; grid on; hold on;

% Карта на фоне
contourf(XG, YG, Total_Heat_Map, 30, 'LineColor', 'none', 'HandleVisibility', 'off');
colormap(flipud(jet));
colorbar; % Добавляем шкалу расстояний

% Стены и Зона эха 
fill3([-10 -8 -8 -10], [-12 -12 0 0], [2 2 2 2], [0.4 0.4 0.4], 'EdgeColor','k', 'HandleVisibility', 'off');
fill3([-12 -10 -10 -12], [-2 -2 0 0], [2 2 2 2], [0.4 0.4 0.4], 'EdgeColor','k', 'HandleVisibility', 'off');
plot3(rect1_x, rect1_y, 3*ones(size(rect1_x)), '--', 'Color', [1 0.7 0.7], 'HandleVisibility', 'off');
plot3(rect2_x, rect2_y, 3*ones(size(rect2_x)), '--', 'Color', [1 0.7 0.7], 'HandleVisibility', 'off');

plot3(traj_log(1,1), traj_log(2,1), 12, 'ko', 'MarkerFaceColor', 'g', 'MarkerSize', 10, 'DisplayName', 'Start');
plot3(xf, yf, 12, 'rx', 'MarkerSize', 15, 'LineWidth', 3, 'DisplayName', 'Goal');

% Инициализация линии пройденного пути
path_h = plot3(traj_log(1,1), traj_log(2,1), 5, 'k:', 'LineWidth', 1.2, 'DisplayName', 'Trajectory');

% Инициализация змеи  
link_h = gobjects(N, 1);
for j = 1:N
    if j == 1
        link_h(j) = plot3(nan, nan, 8, 'r-o', 'LineWidth', 3, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'DisplayName', 'Snake Robot');
    else
        link_h(j) = plot3(nan, nan, 8, 'r-o', 'LineWidth', 3, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
    end
end

% Включаем легенду
legend('Location', 'northeastoutside');

view(2); 
% Устанавливаем общие границы, чтобы видеть всю карту
xlim([min(x_vec) max(x_vec)]);
ylim([min(y_vec) max(y_vec)]);

skip = floor(0.0025/dt); 

for i = 1:skip:length(t_plot)
    if ~ishandle(path_h), break; end
    
    curr_p = traj_log(:, i);
    curr_th = theta_log(:, i);
    
    % Обновление линии пути
    cur_path_x = traj_log(1, 1:i);
    cur_path_y = traj_log(2, 1:i);
    cur_path_z = 5 * ones(size(cur_path_x)); 
    set(path_h, 'XData', cur_path_x, 'YData', cur_path_y, 'ZData', cur_path_z);
    
    % Обновление звеньев змеи
    X_links = -l_half * K_pos * cos(curr_th) + curr_p(1);
    Y_links = -l_half * K_pos * sin(curr_th) + curr_p(2);
    for j = 1:N
        x_s = X_links(j) - l_half * cos(curr_th(j));
        x_e = X_links(j) + l_half * cos(curr_th(j));
        y_s = Y_links(j) - l_half * sin(curr_th(j));
        y_e = Y_links(j) + l_half * sin(curr_th(j));
        set(link_h(j), 'XData', [x_s, x_e], 'YData', [y_s, y_e], 'ZData', [8, 8]);
    end
    
    % % Если нужно слежение камеры
    % xlim([curr_p(1)-2, curr_p(1)+2]);
    % ylim([curr_p(2)-2, curr_p(2)+2]);
    
    title(sprintf('Snake Navigation | Time: %.2f s', t_plot(i)));
    drawnow limitrate;
end

function [phi_basis, dphide] = get_basis_reduced(rho, alpha)
    phi_basis = [rho^2; alpha^2; rho*alpha];
    dphide = [2*rho, 0; 0, 2*alpha; alpha, rho];
end
function out = wrapToPi(angle)
    out = mod(angle + pi, 2*pi) - pi;
end