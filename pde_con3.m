% ����΢�ַ��̵���ֵ���ʵ������֮������(���������Ƕ������ȵ�)
function [c, ceq] = pde_con3(x,T_conv)
    d = x(1); % ����A��ά�ȵ���
    rho_s = x(2); % ����
    rho_w = x(3); % γ��
    theta_s = x(4); % ��ɴ�����Ƕ�
    theta_w = x(5); % γɴ�����Ƕ�
    
    % �����ɵ���A��ά�ȵ��ʵõ���֯�������ȵ��ʺ�����ɢ��
    k_fabric_new = obj_func(d, rho_s, rho_w, theta_s, theta_w); % ֯�������ȵ��� (W/(mK))
    rho_f_c_f = 4.977375565610860e+04;
    
    d0 = 0.6e-3; % ����A��άֱ�� (m)
    H = 50; %֯�����Ķ�������ϵ��(w/(m^2*k))
    p = 1e-3; %hotdisk�㶨����(W)
    h = 2*d0; % ֯���� (m)
    Ta = 25; % �����¶� (��)
    k_a = 0.0296; % �����ȵ��� (W/(mK)) 
    rho_a = 1.184; % �����ܶ� (kg/m^3) 
    c_a = 1012; % �������� (J/(kgK)) 
    % ��ȡ����1�и�����ʵ�����ݣ��õ���Դ�¶���ʱ��仯�ĺ���
    % ��ȡ����1������
    data = xlsread('����1.xlsx');
    t_data = data(:,1); % ʱ������ (s)
    T_data = data(:,2); % �¶����� (��)
    T1 = @(t) interp1(t_data,T_data,t,'linear','extrap'); % ��Դ�¶���ʱ��仯�ĺ��� (��)
    
    % ����ռ��ʱ�������
    x_max = h; % �ռ����ֵ (m)
    x_min = 0; % �ռ���Сֵ (m)
    t_max = 0.1; % ʱ�����ֵ (s)
    t_min = 0; % ʱ����Сֵ (s)
    N_x = 50; % �ռ�������
    N_t = 10000; % ʱ��������
    dx = (x_max - x_min) / N_x; % �ռ䲽�� (m)
    dt = (t_max - t_min) / N_t; % ʱ�䲽�� (s)
    x = x_min:dx:x_max; % �ռ������
    t = t_min:dt:t_max; % ʱ�������
    

    % �������޲�ַ��е���ϵ��
    r3 = k_fabric_new/rho_f_c_f * dt / dx^2; % ֯�ﲿ�ֵ�ϵ��
    r2 = k_a * dt / (rho_a * c_a * dx^2); % �������ֵ�ϵ��
    
    % �����ʼ�����ͱ߽�����
    T0 = Ta * ones(1, N_x + 1); % ��ʼ������֯���ڲ��¶Ⱦ�Ϊ�����¶�
    %����֯������(������Ӵ����)
    A = (p*h)/(0.033*(T1(0.1)-T1(0)));
    %����µı߽�����
    T_left = (p/(H*A))+ T1(t);
    T_right = Ta  * ones(1, N_t + 1); % �ϱ߽�������֯���ϱ����¶�Ϊ�����¶�

    % ʹ����ʽ���޲�ַ����ƫ΢�ַ���ģ�ͣ��õ�֯������¶ȷֲ�
    T_app = zeros(N_t + 1, N_x + 1); % ��ʼ�����¶Ⱦ������ڴ洢ÿ��ʱ��ÿ��λ�õ����¶ȣ���һ��Ϊ��ʼ����
    T_app(1, :) = T0; % ��һ��Ϊ��ʼ����
    for n = 1:N_t % ʱ��ѭ��
        T_app(n + 1, 1) = T_left(n + 1); % �±߽�����
        T_app(n + 1, N_x + 1) = T_right(n + 1); % �ϱ߽�����
        for i = 2:N_x % �ռ�ѭ��
            T_app(n + 1, i) = (r3 + r2) * (T_app(n, i + 1) + T_app(n, i - 1)) + (1 - 2 * r3 - 2 * r2) * T_app(n, i); % ��ʽ���޲�ָ�ʽ
        end
    end

    % ����΢�ַ��̵���ֵ���ʵ������֮�������Ϊ��ʽԼ������
    error_pde = norm(T_conv - T_app, 'fro'); % ʹ��F�����������
    c = []; % û�в���ʽԼ������
    ceq = error_pde - 1e-5; % ��ʽԼ��������Ҫ�����С��0.01
end