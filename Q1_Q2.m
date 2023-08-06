clc;clear;
%%
% 定义已知的参数和常数，如空气热导率、纤维直径、织物厚度、经密、纬密、经纱弯曲角度、纬纱弯曲角度、织物整体热导率等
k_air = 0.0296; % 空气热导率 (W/(mK))
d = 0.6e-3; % 单根A纤维直径 (m)
h = 2*d; % 织物厚度 (m)
rho_s = 60/0.1; % 经密 (根数/m)
rho_w = 80/0.1; % 纬密 (根数/m)
theta_s = 19.8*pi/180; % 经纱弯曲角度 (rad)
theta_w = 25.64*pi/180; % 纬纱弯曲角度 (rad)
k_fabric = 0.033; % 织物整体热导率 (W/(mK))

%%求解单根纤维热导率
% 计算经纱和纬纱部分的体积分数和热导率，将其表示为单根A纤维热导率的函数
V_warp = rho_s*(d^2*pi+8*theta_s*d^2)/(4*h); % 经纱部分体积分数
k_warp = @(k_fiber) k_fiber*k_air /(k_fiber*V_warp+k_air*(1-V_warp)); % 经纱部分热导率，是k_fiber的函数

% 计算纬纱部分的体积分数和热导率，将其表示为单根A纤维热导率的函数
V_weft = rho_w*(d^2*pi+8*theta_w*d^2)/(4*h); % 纬纱部分体积分数
k_weft = @(k_fiber) k_fiber*k_air /(k_fiber*V_weft+k_air*(1-V_weft)); % 纬纱部分热导率，是k_fiber的函数

% 计算织物的整体热导率，将其表示为单根A纤维热导率的函数
k_fabric_func = @(k_fiber) ((k_fiber*k_air /(k_fiber*V_warp+k_air*(1-V_warp)))*(k_fiber*k_air /(k_fiber*V_weft+k_air*(1-V_weft))))/((k_fiber*k_air /(k_fiber*V_warp+k_air*(1-V_warp)))+(k_fiber*k_air /(k_fiber*V_weft+k_air*(1-V_weft)))); % 织物整体热导率，是k_fiber的函数

% 定义一个非线性方程，将织物的整体热导率设为已知值，求解单根A纤维热导率
f = @(k_fiber) k_fabric_func(k_fiber) - k_fabric; % 方程是 f(k_fiber) = 0

% 使用fsolve函数求解非线性方程，得到单根A纤维热导率的数值解
options = optimoptions('fsolve','Display','iter'); % 设置选项为显示迭代过程
k_fiber = fsolve(f,0.1,options); % 使用fsolve函数，初始值设为0.1

% 输出结果，并画出织物的整体热导率与单根A纤维热导率之间的关系曲线
fprintf('单根A纤维的热导率是 %.4f W/(mK)\n',k_fiber);

%纤维热导率可视化
% 画出织物的整体热导率与单根A纤维热导率之间的关系曲线
k_fiber_range = linspace(0.1, 0.2, 300); % 定义一个范围的值给k_fiber
k_fabric_range = arrayfun(@(k_fiber) k_fabric_func(k_fiber), k_fiber_range); % 计算相应的值给k_fabric
plot(k_fiber_range,k_fabric_range,'b-','LineWidth',2); % 画出曲线
hold on;
plot(k_fiber,k_fabric,'r*','MarkerSize',20); % 画出解的点
hold off;
xlabel('单根A纤维的热导率 (W/(mK))'); % 标注x轴
ylabel('织物的整体热导率 (W/(mK))'); % 标注y轴
title('织物的整体热导率与单根A纤维热导率之间的关系'); % 添加标题
grid on; % 打开网格

%%
%分别用材料给的k_fiber和算出来的k_fiber进行数值模拟
alpha = 0.663e-6; % 织物热扩散率 (m^2/s) 
k_a = 0.0296; % 空气热导率 (W/(mK)) 
rho_a = 1.184; % 空气密度 (kg/m^3) 
c_a = 1012; % 空气比热 (J/(kgK)) 
Ta = 25; % 环境温度 (℃)

% 读取附件1中给出的实验数据，得到热源温度随时间变化的函数
% 读取附件1的数据
data = xlsread('附件1.xlsx');
t_data = data(:,1); % 时间数据 (s)
T_data = data(:,2); % 温度数据 (℃)
T1 = @(t) interp1(t_data,T_data,t,'linear','extrap'); % 热源温度随时间变化的函数 (℃)

% 定义织物的密度和比热，根据题目给出的织物整体热扩散率和热导率反求出
rho_f_c_f = k_fabric / alpha ; % 织物比热*密度

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

% 定义有限差分法中的系数
r1 = alpha * dt / dx^2; % 织物部分的系数
r2 = k_a * dt / (rho_a * c_a * dx^2); % 空气部分的系数

% 定义初始条件和边界条件
T0 = Ta * ones(1, N_x + 1); % 初始条件，织物内部温度均为环境温度
T_left = T1(t); % 下边界条件，热源温度随时间变化由附件1给出
T_right = Ta  * ones(1, N_t + 1); % 上边界条件，织物上表面温度为环境温度

% 使用显式有限差分法求解偏微分方程模型，得到织物的温度分布
T_exp = zeros(N_t + 1, N_x + 1); % 初始化温度矩阵，用于存储每个时刻每个位置的温度，第一行为初始条件
T_exp(1, :) = T0; % 第一行为初始条件
for n = 1:N_t % 时间循环
    T_exp(n + 1, 1) = T_left(n + 1); % 下边界条件
    T_exp(n + 1, N_x + 1) = T_right(n + 1); % 上边界条件
    for i = 2:N_x % 空间循环
        T_exp(n + 1, i) = (r1+r2) * (T_exp(n, i + 1) + T_exp(n, i - 1)) + (1-2*r1-2*r2) * T_exp(n, i) ; % 显式有限差分格式
    end
end

% 使用题目给出的织物整体热导率进行数值模拟的结果为实验解，用算出来的k_fiber进行偏微分方程的数值模拟的结果为近似解

% 计算由单根A纤维热导率得到的织物整体热导率和热扩散率
k_fabric_new = k_fabric_func(k_fiber); % 织物整体热导率 (W/(mK))
alpha_new = k_fabric_new / rho_f_c_f; % 织物整体热扩散率 (m^2/s)

% 定义有限差分法中的新系数
r3 = alpha_new * dt / dx^2; % 织物部分的新系数

% 使用显式有限差分法求解偏微分方程模型，得到织物的新温度分布
T_app = zeros(N_t + 1, N_x + 1); % 初始化新温度矩阵，用于存储每个时刻每个位置的新温度，第一行为初始条件
T_app(1, :) = T0; % 第一行为初始条件
for n = 1:N_t % 时间循环
    T_app(n + 1, 1) = T_left(n + 1); % 下边界条件
    T_app(n + 1, N_x + 1) = T_right(n + 1); % 上边界条件
    for i = 2:N_x % 空间循环
        T_app(n + 1, i) = (r3+r2) * (T_app(n, i + 1) + T_app(n, i - 1)) + (1-2*r3-2*r2) * T_app(n, i); % 显式有限差分格式
    end
end

%% 
%可视化部分
%画出时间温度分布图，用题目给的织物整体热扩散率进行模拟的为实验解，用计算得到的织物整体热扩散率，为近似解
figure('Position', [100, 100, 1500, 500]);
% 创建第一个子图，实验解
subplot(1, 2, 1);
surf(x, t, T_exp, 'EdgeColor', 'none'); % 画出实验解的三维曲面图
xlabel('空间位置 x (m)'); % 标注x轴
ylabel('时间 t (s)'); % 标注y轴
zlabel('温度 T (℃)'); % 标注z轴
title('实验解：织物的温度分布'); % 添加标题
colorbar; % 添加色标

% 创建第二个子图，近似解
subplot(1, 2, 2);
surf(x, t, T_app, 'EdgeColor', 'none'); % 画出近似解的三维曲面图
xlabel('空间位置 x (m)'); % 标注x轴
ylabel('时间 t (s)'); % 标注y轴
zlabel('温度 T (℃)'); % 标注z轴
title('近似解：织物的温度分布'); % 添加标题
colorbar; % 添加色标


% 任选一个x的取值，得到其时间-温度的图像，画在同一幅图上进行对比，这里选择x=0.5h作为例子
x_index = round(0.5 * N_x); % 选择x=0.5h对应的网格点索引
T_exp_x = T_exp(:, x_index); % 实验解在x=0.5h处的温度随时间变化的数据
T_app_x = T_app(:, x_index); % 近似解在x=0.5h处的温度随时间变化的数据

figure(3);
plot(t, T_exp_x, 'b-', 'LineWidth', 2); % 画出实验解在x=0.5h处的温度随时间变化的曲线
hold on;
plot(t, T_app_x, 'r--', 'LineWidth', 2); % 画出近似解在x=0.5h处的温度随时间变化的曲线
hold off;
xlabel('时间 t (s)'); % 标注x轴
ylabel('温度 T (℃)'); % 标注y轴
title('实验解和近似解在 x=0.5h 处的温度随时间变化'); % 添加标题
legend('实验解', '近似解'); % 添加图例

% 根据这个任选x的时间-温度的值，画一张实验解-近似解的时间-相对误差图进行展示，相对误差定义为(T_exp-T_app)/T_exp*100%
error_x = abs(T_exp_x - T_app_x) ./ T_exp_x * 100; % 计算相对误差百分比

figure(4);
plot(t, error_x, 'k-', 'LineWidth', 2); % 画出相对误差随时间变化的曲线
xlabel('时间 t (s)'); % 标注x轴
ylabel('相对误差 (%)'); % 标注y轴
title('实验解和近似解在 x=0.5h 处的相对误差随时间变化'); % 添加标题

%%
%第二问，寻找最优参数
% 定义目标函数、变量、约束条件和选项等参数 
obj_func_new = @(x) (k_fiber*k_air) ./ (k_fiber*(x(3)*(pi/4 + 2*x(5))*x(1)^2)/h + k_air*(1-(x(3)*(pi/4 + 2*x(5))*x(1)^2)/h) + k_fiber*(x(2)*(pi/4 + 2*x(4))*x(1)^2)/h + k_air*(1-(x(2)*(pi/4 + 2*x(4))*x(1)^2)/h)); % 目标函数，是直径、经密、纬密、经纱弯曲角度、纬纱弯曲角度这五个变量的函数
x0 = [3e-4, 40/0.1, 40/0.1, 19.8*pi/180, 25.64*pi/180]; % 变量的初始值
lb = [3e-4, 40/0.1, 40/0.1, 10*pi/180, 10*pi/180]; % 变量的下界
ub = [6e-4, 600/0.1, 600/0.1, 26.565*pi/180, 26.565*pi/180]; % 变量的上界
nonlcon = @(x) pde_con(x,T_exp); % 非线性约束，是一个函数，用于计算微分方程的数值解和实验数据之间的误差
options = optimoptions('simulannealbnd','Display','iter','MaxIterations',10000,'PlotFcn',{@saplotbestx, @saplotbestf, @saplotx, @saplotf}); % 设置选项为显示迭代过程，并且定义一个输出函数

% 调用 simulannealbnd 函数求解优化问题
[x_opt,f_opt] = simulannealbnd(obj_func_new,x0,lb,ub,options); % 得到目标函数的最小值和最优解

%%options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm','active-set', 'TolFun', 1e-6, 'TolCon', 1e-6); % 设置选项为显示迭代过程，使用有效集算法，设置目标函数和约束条件的容忍度

% 调用 fmincon 函数求解优化问题
%%[x_opt, f_opt] = fmincon(obj_func_new, x0, [], [], [], [], lb, ub, nonlcon, options); % 得到目标函数的最小值和最优解


% 输出结果，并画出目标函数在不同变量取值下的曲面图或等高线图
fprintf('使得织物整体热导率最低时对应的直径是 %.4f m \n', x_opt(1));
fprintf('使得织物整体热导率最低时对应的经密是 %.2f 根数/m\n', x_opt(2));
fprintf('使得织物整体热导率最低时对应的纬密是 %.2f 根数/m\n', x_opt(3));
fprintf('使得织物整体热导率最低时对应的经纱弯曲角度是 %.2f 度\n', x_opt(4)*180/pi);
fprintf('使得织物整体热导率最低时对应的纬纱弯曲角度是 %.2f 度\n', x_opt(5)*180/pi);
fprintf('使得织物整体热导率最低时对应的目标函数值是 %.4f W/(mK)\n', f_opt);