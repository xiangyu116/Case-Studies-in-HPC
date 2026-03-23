%c monitor
function [x,resvec,iter]=gmres_monitor(A,b,maxit,tol,x0)
% Input:
%   A-Coefficient matrix (n×n)
%   b-Right-hand side vector (n×1)
%   maxit-Maximum number of iterations
%   tol-Convergence tolerance
%   x0-Initial guess solution (optional)
% Output:
%   x-Approximate solution
%   resvec-History of residual norms
%   iter-Actual number of iterations
n=length(b);
if nargin<5
    x0=zeros(n,1);
end
r0=b-A*x0;
beta=norm(r0);
x=x0;
resvec=beta;
iter=0;
if beta<tol
    return;
end
V=zeros(n, maxit+1);
H=zeros(maxit+1, maxit);
cs=zeros(maxit,1); % Store the cosine value of the Givens rotation
sn=zeros(maxit,1); % Store the sine value of the Givens rotation
g=zeros(maxit+1,1); % Store the rotated right end vector

V(:,1)=r0/beta;
g(1)=beta;
iter=maxit;
for j=1:maxit
    w=A*V(:,j);
    for i=1:j
        H(i,j)=V(:,i)'*w;
        w=w-H(i,j)*V(:,i);
    end
    H(j+1,j)=norm(w);
    if H(j+1,j) ~= 0
        V(:,j+1) = w/H(j+1,j);
    end
    % Apply the previously stored Givens rotation to the first j-1 columns
    for i=1:j-1
        [H(i,j),H(i+1,j)]=apply_givens(cs(i),sn(i),H(i,j),H(i+1,j));
    end
    [cs(j),sn(j)]=givens_rotation(H(j,j),H(j+1,j));
    [H(j,j),H(j+1,j)]=apply_givens(cs(j),sn(j),H(j,j),H(j+1,j));
    [g(j), g(j+1)]=apply_givens(cs(j),sn(j),g(j),g(j+1));
    res = abs(g(j+1));
    resvec(end+1) = res;
    
    if res<tol
        iter=j;
        break;
    end
end

y=H(1:iter,1:iter)\g(1:iter);
x=x0+V(:,1:iter)*y;

end