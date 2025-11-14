%% IMPACT4Mech - Continuous-Time Stabilization from Noisy Data
% Numerical example 2 of:
% A. Bosso, M. Borghesi, A. Iannelli, B. Yi, G. Notarstefano,
% "Data-Driven Stabilization of Continuous-Time LTI Systems from
% Noisy Input–Output Data"

% This file requires the installation of MOSEK and YALMIP
% MOSEK:  https://docs.mosek.com/10.2/toolbox/index.html
% YALMIP: https://yalmip.github.io

%% Startup functions

clear
clc

% plot settings
% fonts
font_size   = 15;
% lines
line_width  = 1;
% plot position
plot_x      = 500;
plot_y      = 300;
% plot size
plot_width  = 500;
plot_height = 200;

%% System definition

% dimensions
m   = 2;
p   = 2;
n   = 2;

% input-output model
A_0 = [ -20.97  -48.63;
         2.643   5.867];
A_1 = [  5.297  -10.47;
       -0.2764   6.371];
B_0 = [ -59.44  -12.63;
         12.59  0.8696];
B_1 = [      0  -3.146;
         5.679       0];
E_0 =   eye(2);
E_1 = zeros(2);

% state-space realization
A = [zeros(2)  -A_0;
       eye(2)  -A_1];
B = [B_0;
     B_1];
E = [E_0;
     E_1];
C = [zeros(2)  eye(2)];

%% Simulation parameters

% experiment duration
T  = 3;
dt = T/1000000;
t  = 0:dt:T;

% plant initial conditions
x0 = zeros(n*p, 1);

% applied input
omega1 = 5;
omega2 = 7;
u      = [5*sin(1*omega1*t) + 4*sin(2*omega2*t) +...
          3*sin(3*omega1*t) + 2*sin(4*omega2*t) +...
          1*sin(1*omega1*t);
          5*sin(1*omega2*t) + 4*sin(2*omega1*t) +...
          3*sin(3*omega2*t) + 2*sin(4*omega1*t) +...
          1*sin(1*omega2*t)];

% process noise
N_modes = 100;

Phi = zeros(2*N_modes + 1, length(t));
Phi(1, :) = 1/sqrt(T);
for l = 1:N_modes
    Phi(2*l, :)     = sqrt(2/T)*cos(2*l*pi*t/T);
    Phi(2*l + 1, :) = sqrt(2/T)*sin(2*l*pi*t/T);
end

%% Algorithm parameters

% controller dimension
mu = n*(m + p);

% filter gains
Lambda = [ 0 -12;
           1  -7];
Gamma  = [ 0;  1];

% observer matrices
F = kron(eye(p + m), Lambda);
G = [zeros(n*p, m); kron(eye(m), Gamma)];
L = [kron(eye(p), Gamma); zeros(n*m, p)];

% L_2[0, T] gain
gamma          = 0.07685;
% simulation in backward time
[t_sim, W_sim] = ode45(@(t_sim, W_vec) ...
                  DRE(t_sim, W_vec, kron(Lambda, eye(p)), E, C, gamma), ...
                       [0 T], zeros((n*p)^2, 1));
% gamma certified if W_sim exists over [0, T]

%% Simulations

N_runs    = 200;
delta_w   = [2.5 3 3.4 3.7 3.9];
LMI_test  = zeros(size(delta_w));
rho       = zeros(N_runs, length(delta_w));

for j = 1:length(delta_w)

    disp('delta_w:')
    disp(delta_w(j))
    delta = gamma^2*delta_w(j);
    Delta = delta*eye(p);
    
for k = 1:N_runs

    disp('Test #:')
    disp(k)
    
    %% Dataset generation

    % process noise generation
    alpha   = randn(4*N_modes + 2, 1);
    alpha   = sqrt(delta_w(j))*alpha/(max(norm(alpha), 1e-12));

    alpha1 = alpha(1:2*N_modes + 1);
    alpha2 = alpha(2*N_modes + 2:end);

    w1 = alpha1'*Phi;
    w2 = alpha2'*Phi;
    disp('Energy of w:')
    disp(sum(w1.^2)*dt + sum(w2.^2)*dt)
    w  = [w1; w2];

    % plant simulation
    plant = ss(A, [B E], C, []);
    y     = lsim(plant, [u; w], t, x0)';

    %% Filtering

    aux = ss(Lambda, Gamma, eye(n), []);
    chi = lsim(aux, zeros(size(t)), t, Gamma)';
    obs = ss(F, [L G], eye(mu), []);
    z   = lsim(obs, [y; u], t, zeros(mu, 1))';

    y_zeta = [y; -chi; -z];

    Zeta = zeros(p + n + mu, p + n + mu);
    for i = 2:size(t, 2)
        Zeta = Zeta + (1/2)*dt*(y_zeta(:,   i)*y_zeta(:,   i)' + ...
                                y_zeta(:, i-1)*y_zeta(:, i-1)');
    end

    Z = Zeta(p+1:end, p+1:end);

    N   = blkdiag(Delta, zeros(n + mu)) - Zeta;
    N_L = blkdiag(L, eye(n + mu))*N*blkdiag(L', eye(n + mu));

    %% Computing the control gain

    % decision variables
    P = sdpvar(mu, mu);
    Q = sdpvar(m, mu);

    % LMI constraints
    P_LMI = P >= 1e-6;
    S_LMI = -[F*P + P'*F' + G*Q + Q'*G'   zeros(mu, n)            P';
                           zeros(n, mu)       zeros(n)  zeros(n, mu);
                                      P   zeros(mu, n)     zeros(mu)] ...
            - N_L >= 1e-6;

    % solving the LMI
    constr = P_LMI + S_LMI;
    obj    = 0;
    ops    = sdpsettings('solver', 'mosek', 'verbose', 0);
    diagnostics = optimize(constr, obj, ops);

    P = value(P);
    Q = value(Q);
    K =   Q*P^-1;

    %% Rho

    disp('rho:')
    ratio     = delta/(min(real(eig(Z))));
    disp(ratio)
    rho(k, j) = ratio;

    %% Stability verification

    % controller matrices
    A_c = F + G*K;
    B_c = L;
    C_c = K;
    D_c = zeros(m, p);

    % closed-loop matrix
    A_cl = [A + B*D_c*C  B*C_c;
                  B_c*C    A_c];

    if diagnostics.problem == 0
        if max(real(eig(A_cl))) < 0
            disp('Stabilization successful')
            LMI_test(j) = LMI_test(j) + 1;
        else
            error('The LMI did not return a stabilizing controller')
        end
    else
        disp('Unfeasible LMI')
    end
end
end

%% Plots

figure(1)
% left axis
yyaxis left
box_h = boxplot(rho, delta_w, 'Labels', string(delta_w));
set(findobj(box_h, 'Type', 'Line'), 'LineWidth', line_width);
hold on
box on
grid on
ylabel('$\rho$', FontSize=font_size, Interpreter='latex')
ax = gca;
ax.YColor = 'k';
% right axis
yyaxis right
plot_h = plot(1:length(delta_w), 100*LMI_test/N_runs, '-o', 'LineWidth', 1.8, ...
     'Color', [0.2 0.6 0.2], 'MarkerFaceColor', [0.2 0.6 0.2]);
plot_h.Clipping = 'off';
ylabel('Feasibility rate (%)', FontSize=font_size)
ax.YColor = [0.2 0.6 0.2];
% common parts
xlabel('$\delta_w$', FontSize=font_size, Interpreter='latex')

set(gcf,'position', [plot_x, plot_y, plot_width, plot_height])

%% Riccati Differential Equation
% vector field in backward time

function dW_vec = DRE(~, W_vec, tLambda, E, C, gamma)
    np = size(tLambda, 1);
    W  = reshape(W_vec, np, np);
    dW = tLambda'*W + W*tLambda + gamma^(-2)*W*(E*E')*W + C'*C;
    dW = (dW + dW')/2;
    dW_vec = dW(:);
end