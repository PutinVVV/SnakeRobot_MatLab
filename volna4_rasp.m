clear; clc; close all;

% ПАРАМЕТРЫ СРЕДЫ
xf = -10; yf = -4;     % Цель 
x0 = 0; y0 = 0;        % Старт
margin = 0.6;          % Зона эха

% Параметры сетки
grid_res = 0.3; 
x_vec = -15:grid_res:2;
y_vec = -6:grid_res:6;
[XG, YG] = meshgrid(x_vec, y_vec);

% стены
obs_mask = (XG >= -5 & XG <= -3 & YG >= -2 & YG <= 6) | ... 
           (XG >= -9 & XG <= -7 & YG >= -6 & YG <= 2);

Total_Heat_Map = Inf(size(XG));
[~, iy_f] = min(abs(y_vec - yf));
[~, ix_f] = min(abs(x_vec - xf));
Total_Heat_Map(iy_f, ix_f) = 0;

queue = [iy_f, ix_f];
dirs = [1,0; -1,0; 0,1; 0,-1; 1,1; 1,-1; -1,1; -1,-1];
d_costs = [1, 1, 1, 1, sqrt(2), sqrt(2), sqrt(2), sqrt(2)] * grid_res;

vis_skip = 10;   
iter = 0;
wait_s = 3000;   

figure('Name','Wavefront Animation (Contour Style)','Position', [100 100 850 600]);

fprintf('start...\n');

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
    
    iter = iter + 1;
    
    if mod(iter, vis_skip) == 0 || isempty(queue)
        cla; 
        display_map = Total_Heat_Map;
        display_map(isinf(display_map)) = 25; 
        
        contourf(XG, YG, display_map, 30, 'LineColor', 'none'); 
        hold on; colormap(flipud(jet)); colorbar;
        caxis([0 25]); 
         %  препятствия
        fill([-5 -3 -3 -5], [-2 -2 6 6], [0.4 0.4 0.4], 'EdgeColor', 'k');
        fill([-9 -7 -7 -9], [-6 -6 2 2], [0.4 0.4 0.4], 'EdgeColor', 'k');

        plot(x0, y0, 'ko', 'MarkerFaceColor','g', 'MarkerSize',8);
        plot(xf, yf, 'rx', 'MarkerSize',12, 'LineWidth',3);

        xlabel('x [m]'); ylabel('y [m]'); 
        title(['Расчет тепловой карты. Итерация: ', num2str(iter)]);
        axis equal; grid on;
        
        drawnow limitrate; 
    end
    
    % Условие паузы
    if iter == wait_s
        pause;
    end
end
