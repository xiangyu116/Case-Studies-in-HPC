%a basic
function [x,resvec]=gmres_basic(A,b,maxit,tol,x0)
%   A-Coefficient matrix
%   b-Right-hand side vector
%   maxit-Maximum number of iterations
%   tol-Convergence tolerance
%   x0-Initial guess solution (optional)
% Output:
%   x-Approximate solution
%   resvec-History of residual norms
n=length(b);
if nargin<5
    x0=zeros(n,1);
end
r0=b-A*x0;  % Calculate the initial residual
beta=norm(r0);  % Calculate the initial residual norm
x=x0;  % Initialization solution
resvec=beta;  % Initialize residual record

if beta<tol
    return;  % If the initial residual has met the tolerance, simply return.
end

V=zeros(n,maxit+1);  % Initialize the Krylov subspace basis vector matrix
H=zeros(maxit+1,maxit);  % Initialize the Hessenberg matrix
V(:,1)=r0/beta; % Normalize the initial residual

for j=1:maxit
    w=A*V(:,j);
    for i=1:j
        H(i,j)=V(:,i)'*w;
        w=w-H(i,j)*V(:,i);
    end
    H(i+1,j)=norm(w);  % Calculate the elements on the secondary diagonal
    if H(j+1,j)~=0
        V(:,j+1)=w/H(j+1,j);
    end
    e1=zeros(j+1,1);
    e1(1)=beta;
    Hj=H(1:j+1,1:j);  % Extract the current Hessenberg submatrix
    y=Hj\e1;
    x=x0+V(:,1:j)*y;
    res=norm(b-A*x);
    resvec(end+1)=res;  % Record residuals
    if res < tol
        return;
    end
end
end