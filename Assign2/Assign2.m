%test
clear;clc;close all;

tol=1e-8;
maxit=200; % Maximum number of iterations

% small: cdde6

load('cdde6.mat');

A_small=Problem.A;  % Obtain the coefficient matrix
n_small=size(A_small,1);  % Obtain the dimensions of the matrix
b_small=ones(n_small,1);  % Set the right end vector to a vector of all 1s

fprintf('Small matrix: %s, size=%d\n', Problem.name, n_small);

[x1,r1]=gmres_basic(A_small, b_small, maxit, tol);  %basic
[x2,r2]=gmres_givens(A_small, b_small, maxit, tol);  %givens
[x3,r3,it3]=gmres_monitor(A_small, b_small, maxit, tol);  %monitor

fprintf('basic residua=%.3e\n', norm(b_small-A_small*x1));
fprintf('givens residual=%.3e\n', norm(b_small-A_small*x2));
fprintf('monitor residual=%.3e, iter=%d\n', norm(b_small-A_small*x3), it3);


figure;
semilogy(0:length(r1)-1, r1, '-o', 'LineWidth', 1.2); 
hold on;
semilogy(0:length(r2)-1, r2, '-s', 'LineWidth', 1.2);
semilogy(0:length(r3)-1, r3, '-^', 'LineWidth', 1.2);
xlabel('Iteration');
ylabel('Residual norm');
title('GMRES on cdde6 (small)');
legend('basic','givens','monitor','Location','best');
grid on;

%large
load('parabolic_fem.mat');
A_large_full=Problem.A;

A_large=A_large_full;
n_large=size(A_large,1);
b_large=ones(n_large,1);
fprintf('\nLarge matrix: %s, size=%d\n', Problem.name, n_large);

[x4,r4]=gmres_basic(A_large,b_large,maxit,tol);  %basic
[x5,r5]=gmres_givens(A_large,b_large,maxit,tol);  %givens
[x6,r6,it6]=gmres_monitor(A_large,b_large,maxit,tol);  %monitor

fprintf('basic residual=%.3e\n', norm(b_large-A_large*x4));
fprintf('givens residual=%.3e\n', norm(b_large-A_large*x5));
fprintf('monitor residual=%.3e, iter=%d\n', norm(b_large-A_large*x6),it6);

figure;
semilogy(0:length(r4)-1, r4, '-o', 'LineWidth', 1.2); 
hold on;
semilogy(0:length(r5)-1, r5, '-s', 'LineWidth', 1.2);
semilogy(0:length(r6)-1, r6, '-^', 'LineWidth', 1.2);
xlabel('Iteration');
ylabel('Residual norm');
title('GMRES on parabolic\_fem (large)');
legend('basic','givens','monitor','Location','best');
grid on;