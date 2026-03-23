%apply_givens
function [v1_new,v2_new]=apply_givens(c,s,v1,v2)
v1_new=c*v1+s*v2;
v2_new=-s*v1+c*v2;
end