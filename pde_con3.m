% 计算微分方程的数值解和实验数据之间的误差(问题三考虑对流换热的)
function [c, ceq] = pde_con3(x,T_conv)
    d = x(1); % 单根A纤维热导率
    rho_s = x(2); % 经密
    rho_w = x(3); % 纬密
    theta_s = x(4); % 经纱弯曲角度
    theta_w = x(5); % 纬纱弯曲角度
    
    % 计算由单根A纤维热导率得到的织物整体热导率和热扩散率
    k_fabric_new = obj_func(d, rho_s, rho_w, theta_s, theta_w); % 织物整体热导率 (W/(mK))
    rho_f_c_f = 4.977375565610860e+04;
    
    d0 = 0.6e-3; % 单根A纤维直径 (m)
    H = 50; %织物表面的对流换热系数(w/(m^2*k))
    p = 1e-3; %hotdisk恒定功率(W)
    h = 2*d0; % 织物厚度 (m)
    Ta = 25; % 环境温度 (℃)
    k_a = 0.0296; % 空气热导率 (W/(mK)) 
    rho_a = 1.184; % 空气密度 (kg/m^3) 
    c_a = 1012; % 空气比热 (J/(kgK)) 
    % 读取附件1中给出的实验数据，得到热源温度随时间变化的函数
    % 读取附件1的数据
    data = xlsread('附件1.xlsx');
    t_data = data(:,1); % 时间数据 (s)
    T_data = data(:,2); % 温度数据 (℃)
    T1 = @(t) interp1(t_data,T_data,t,'linear','extrap'); % 热源温度随时间变化的函数 (℃)
    
    % 定义空间和时间的网格
    x_max = h; % 空间最大值 (m)
    x_min = 0; % 空间最小值 (m)
    t_max = 0.1; % 时间最大值 (s)
    t_min = 0; % 时间最小值 (s)
    N_x = 50; % 空间网格数
    N_t = 10000; % 时间网格数
    dx = (x_max - x_min) / N_x; % 空间步长 (m)
    dt = (t_max - t_min) / N_t; % 时间步长 (s)
    x = x_min:dx:x_max; % 空间网格点
    t = t_min:dt:t_max; % 时间网格点
    

    % 定义有限差分法中的新系数
    r3 = k_fabric_new/rho_f_c_f * dt / dx^2; % 织物部分的系数
    r2 = k_a * dt / (rho_a * c_a * dx^2); % 空气部分的系数
    
    % 定义初始条件和边界条件
    T0 = Ta * ones(1, N_x + 1); % 初始条件，织物内部温度均为环境温度
    %计算织物表面积(与空气接触面积)
    A = (p*h)/(0.033*(T1(0.1)-T1(0)));
    %求出新的边界条件
    T_left = (p/(H*A))+ T1(t);
    T_right = Ta  * ones(1, N_t + 1); % 上边界条件，织物上表面温度为环境温度

    % 使用显式有限差分法求解偏微分方程模型，得到织物的新温度分布
    T_app = zeros(N_t + 1, N_x + 1); % 初始化新温度矩阵，用于存储每个时刻每个位置的新温度，第一行为初始条件
    T_app(1, :) = T0; % 第一行为初始条件
    for n = 1:N_t % 时间循环
        T_app(n + 1, 1) = T_left(n + 1); % 下边界条件
        T_app(n + 1, N_x + 1) = T_right(n + 1); % 上边界条件
        for i = 2:N_x % 空间循环
            T_app(n + 1, i) = (r3 + r2) * (T_app(n, i + 1) + T_app(n, i - 1)) + (1 - 2 * r3 - 2 * r2) * T_app(n, i); % 显式有限差分格式
        end
    end

    % 计算微分方程的数值解和实验数据之间的误差，作为等式约束条件
    error_pde = norm(T_conv - T_app, 'fro'); % 使用F范数计算误差
    c = []; % 没有不等式约束条件
    ceq = error_pde - 1e-5; % 等式约束条件，要求误差小于0.01
end