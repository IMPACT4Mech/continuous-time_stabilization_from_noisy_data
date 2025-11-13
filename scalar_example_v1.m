%% IMPACT4Mech - Continuous-Time Stabilization from Noisy Data
% Numerical example 1 of:
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
line_width  = 2;
% plot position
plot_x      = 500;
plot_y      = 300;
% plot size
plot_width  = 500;
plot_height = 200;

%% System definition

% dimensions
m   = 1;
p   = 1;
n   = 1;

% input-output model
A_0 = -1;
B_0 =  1;
E_0 =  1;

% state-space realization
A = -A_0;
B =  B_0;
E =  E_0;
C =    1;

%% Continuous-time dataset

% experiment duration
T  = 1;
dt = T/1000000;
t  = 0:dt:T;

% plant initial conditions
x0 = 0;

% applied input
u  = sin(5*pi*t);

% noise signals
load('w_noise.mat')
load('v_noise.mat')

disp('Energy of w:')
disp(sum(w.^2)*dt)
disp('Energy of v:')
disp(sum(v.^2)*dt)

% plant simulation
plant = ss(A, [B E], C, []);
y     = lsim(plant, [u; w], t, x0)' + v;

%% Algorithm parameters

% controller dimension
mu = n*(m + p);

% filter gains
Lambda = -2;
Gamma  =  2;

% observer matrices
F = kron(eye(p + m), Lambda);
G = [zeros(n*p, m); kron(eye(m), Gamma)];
L = [kron(eye(p), Gamma); zeros(n*m, p)];

% L_2[0, T] gain
gamma          = 0.33;
% simulation in backward time
[t_sim, W_sim] = ode45(@(t_sim, W_vec) ...
                       DRE(t_sim, W_vec, Lambda, E, C, gamma), ...
                       [0 T], zeros((n*p)^2, 1));
% gamma certified if W_sim exists over [0, T]

% noise energy bounds
delta_w = 0.0008;
delta_v = 0.0003;

% noise bound computation
Delta = (gamma*sqrt(delta_w) + sqrt(delta_v))^2;
disp('Delta:')
disp(Delta)

%% Filtering

aux = ss(Lambda, Gamma, eye(n), []);
chi = lsim(aux, zeros(size(t)), t, Gamma)';
obs = ss(F, [L G], eye(mu), []);
z   = lsim(obs, [y; u], t, zeros(mu, 1))';

y_zeta = [y; -chi; -z];

Zeta = zeros(p + n + mu, p + n + mu);
for i = 2:numel(t)
    Zeta = Zeta + (1/2)*dt*(y_zeta(:,   i)*y_zeta(:,   i)' + ...
                            y_zeta(:, i-1)*y_zeta(:, i-1)');
end

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
ops    = sdpsettings('solver', 'mosek');
optimize(constr, obj, ops);

P = value(P);
Q = value(Q);
K =   Q*P^-1;

disp('Stabilizing gain K:')
disp(K)
format shortEng
disp('Matrix P:')
disp(P)
format short

%% Stability verification

% controller matrices
A_c = F + G*K;
B_c = L;
C_c = K;
D_c = 0;

% closed-loop matrix
A_cl = [A + B*D_c*C  B*C_c;
              B_c*C    A_c];

disp('Closed-loop eigenvalues:')
disp(eig(A_cl))

%% Non-minimal realization parameters

Pi         = [1.5 0.5];
H          =      C*Pi;
Theta_star = [  0   H];

%% Ellipsoid surface

X = Zeta(p+1:end,     1:p);
Y = Zeta(    1:p,     1:p);
Z = Zeta(p+1:end, p+1:end);

S_N       = Delta - Y + X'*Z^-1*X;
Theta_hat = -X'*Z^-1;

[V, D]    = eig(S_N*Z^-1);
radii     = sqrt(diag(D));

[Th1_E, Th2_E, Th3_E] = ellipsoid(       0,        0,        0, ...
                                  radii(1), radii(2), radii(3), 25);
E_points = V*[Th1_E(:)'; Th2_E(:)'; Th3_E(:)'];

Th1_E = reshape(E_points(1, :), size(Th1_E)) + Theta_hat(1);
Th2_E = reshape(E_points(2, :), size(Th2_E)) + Theta_hat(2);
Th3_E = reshape(E_points(3, :), size(Th3_E)) + Theta_hat(3);

%% Region stabilized by (P, K)

n_grid   = 15;
th1_vals = linspace(-0.033, 0.023, ...
                    n_grid);
th2_vals = linspace(   1.2,     2, ...
                    n_grid);
th3_vals = linspace(   0.1,   0.8, ...
                    n_grid);

[Th1_C, Th2_C, Th3_C] = ndgrid(th1_vals, th2_vals, th3_vals);
max_eig               = zeros(size(Th1_C));
for i   = 1:numel(Th1_C)
    H_i = [Th2_C(i) Th3_C(i)];
    M_i = (F + G*K + L*H_i)*P + P*(F + G*K + L*H_i)';
    max_eig(i) = max(real(eig(M_i)));
end

%% Plots

% signals plot
figure(1)
plot(t, y, 'LineWidth', line_width)
hold on
grid on
box on
plot(t, w, 'LineWidth', line_width)
plot(t, v, 'LineWidth', line_width)
xlabel('$t$ [s]', ...
       FontSize=font_size, Interpreter='latex')
legend('$y$', '$w$', '$v$', ...
       FontSize=font_size, Location='northwest', Interpreter='latex')
set(gcf, 'position', [plot_x, plot_y, plot_width, plot_height])

% ellipsoid plot
figure(2)
mesh(Th1_E, Th2_E, Th3_E, ...
     'EdgeColor', [0 0.3 0.8], 'FaceColor', 'none', 'LineWidth', 0.7);
hold on
grid on
box on
plot3(Theta_star(1), Theta_star(2), Theta_star(3), 'r.', 'MarkerSize', 18);
plot3( Theta_hat(1),  Theta_hat(2),  Theta_hat(3), 'b.', 'MarkerSize', 18);
patch_h = patch(isosurface(Th1_C, Th2_C, Th3_C, max_eig, 0));
set(patch_h, 'FaceColor', [  1  0.6  0.2]);
set(patch_h, 'EdgeColor', [0.8 0.25 0.05]);
Xtri = [th1_vals(end), th1_vals(end), th1_vals(end)];
Ytri = [      1.41183, th2_vals(end), th2_vals(end)];
Ztri = [  th3_vals(1),      0.237072,   th3_vals(1)];
patch(Xtri, Ytri, Ztri, [1 0.8 0.4], ...
                'EdgeColor', 'none', ...
             'FaceLighting', 'none');
xlabel('$\Theta_1$', FontSize=font_size, Interpreter='latex')
ylabel('$\Theta_2$', FontSize=font_size, Interpreter='latex')
zlabel('$\Theta_3$', FontSize=font_size, Interpreter='latex')
xlim([-0.033, 0.023])
ylim([   1.2,     2])
zlim([   0.1,   0.8])
view(68, 7);
set(gcf,'position', [plot_x, plot_y, plot_width, 1.5*plot_height])

%% Riccati Differential Equation
% vector field in backward time

function dW_vec = DRE(~, W_vec, tLambda, E, C, gamma)
    np = size(tLambda, 1);
    W  = reshape(W_vec, np, np);
    dW = tLambda'*W + W*tLambda + gamma^(-2)*W*(E*E')*W + C'*C;
    dW = (dW + dW')/2;
    dW_vec = dW(:);
end