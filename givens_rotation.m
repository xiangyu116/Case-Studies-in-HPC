%d rotation
function [c,s]=givens_rotation(a,b)
% This function is used to construct a planar rotation matrix to rotate the vector [a; b] to [r; 0]
% Input:
%   a-The first component of the vector
%   b-The second component of the vector
% Output:
%   c-The cosine value of the rotation (cos)
%   s-The sine value of the rotation (sin)
if b==0
    % No need to rotate
    c=1;
    s=0;
elseif a==0
    % Rotate 90 degrees
    c=0;
    s=1;
else
    % Rotate according to the formula
    r=hypot(a,b);
    c=a/r;
    s=b/r;
end
end