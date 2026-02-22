%1
% TSQR (4 blocks) demo
% Set the random number seed to 0.
rng(0);

m=4000;   % tall
n=50;     % skinny
A=randn(m, n);

% Ensure m divisible by 4 for simple block splitting
assert(mod(m,4)==0, "m must be divisible by 4 for this simple implementation.");

function [Q, R]=tsqr_4blocks(A)
    [m,n]=size(A);
    assert(mod(m,4)==0, "m must be divisible by 4.");
    b=m/4;

    % 1) Split A into 4 row blocks
    A1=A(1:b, :);
    A2=A(b+1:2*b, :);
    A3=A(2*b+1:3*b, :);
    A4=A(3*b+1:4*b, :);

    % 2) Local (block) QR factorizations (economy/reduced QR)
    [Q1, R1]=qr(A1, 0);   % Q1: b x n, R1: n x n
    [Q2, R2]=qr(A2, 0);
    [Q3, R3]=qr(A3, 0);
    [Q4, R4]=qr(A4, 0);

    % 3) Tree reduction on the small R factors
    [Q12, R12]=qr([R1; R2], 0);  % Q12: (2n) x n, R12: n x n
    [Q34, R34]=qr([R3; R4], 0);

    [Q1234, R]=qr([R12; R34], 0); % Q1234: (2n) x n, R: n x n

    % 4) Build the global Q
    Q_blk=blkdiag(Q1, Q2, Q3, Q4);      % m x (4n)
    Q_tree1=blkdiag(Q12, Q34);            % (4n) x (2n)

    Q=Q_blk*Q_tree1*Q1234;            % (m x n)
end


[Q, R]=tsqr_4blocks(A);

% Factorization accuracy
rel_err=norm(A-Q*R, 'fro')/norm(A, 'fro');

% Orthogonality of Q (should be close to I)
orth_err=norm(Q'*Q-eye(size(Q,2)), 'fro');

fprintf("Relative factorization error ||A-QR||_F / ||A||_F = %.3e\n", rel_err);
fprintf("Orthogonality error ||Q^TQ-I||_F = %.3e\n", orth_err);
