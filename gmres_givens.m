%b given
function [x,resvec]=gmres_givens(A,b,maxit,tol,x0)
% Input:
%   A-Coefficient matrix
%   b-Right-hand side vector
%   maxit-Maximum number of iterations
%   tol-Convergence tolerance
%   x0-Initial guess solution
% Output:
%   x-Approximate solution
%   resvec-History of residual norms
n=length(b);
if nargin<5
    x0=zeros(n,1);
end
r0=b-A*x0; % Calculate the initial residual
beta=norm(r0); % Calculate the initial residual norm
x=x0;
resvec=beta;
if beta<tol
    return;
end
V=zeros(n, maxit+1); % Initialize the Krylov subspace basis vector matrix
H=zeros(maxit+1, maxit); % Initialize the Hessenberg matrix
cs=zeros(maxit,1); % Store the cosine value of the Givens rotation
sn=zeros(maxit,1); % Store the sine value of the Givens rotation
g=zeros(maxit+1,1); % Store the rotated right end vector

V(:,1)=r0/beta;
g(1)=beta;

kfinal=maxit; % Record the final number of iterations
for j=1:maxit
    w=A*V(:,j);
    for i=1:j
        H(i,j)=V(:,i)'*w; % Calculate the elements (projections) of the Hessenberg matrix
        w=w-H(i,j)*V(:,i); % Orthogonalization
    end
    H(j+1,j)=norm(w);
    if H(j+1,j)~=0
        V(:,j+1)=w/H(j+1,j);
    end
    for i=1:j-1
        [H(i,j),H(i+1,j)]=apply_givens(cs(i),sn(i),H(i,j),H(i+1,j));
    end
    % Calculate the new Givens rotation parameters to eliminate the elements on the sub-diagonal
    [cs(j), sn(j)]=givens_rotation(H(j,j),H(j+1,j));
    [H(j,j), H(j+1,j)]=apply_givens(cs(j),sn(j),H(j,j),H(j+1,j));
    % Apply the same rotation to the right-end vector g
    [g(j), g(j+1)]=apply_givens(cs(j),sn(j),g(j),g(j+1));

    res=abs(g(j+1));
    resvec(end+1)=res;
    if res<tol
        kfinal=j;
        break;
    end
end

    y=H(1:kfinal,1:kfinal)\g(1:kfinal);
    x=x0+V(:,1:kfinal)*y; % Update the solution vector
end
