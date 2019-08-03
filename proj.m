% write function that projects the  point (q = X,Y) on a vector
% which is composed of two points - vector = [p0x p0y; p1x p1y]. 
% i.e. vector is the line between point p0 and p1. 
%
% The result is a point qp = [x y] and the length [length_q] of the vector drawn 
% between the point q and qp . This resulting vector between q and qp 
% will be orthogonal to the original vector between p0 and p1. 
%
% vector = [1,1; 4,4];
% q = [3,2];
% I used above values for calling function
function [ProjPoint] = proj(vector, q)
p0 = vector(1,:);
p1 = vector(2,:);
a = [-q(1)*(p1(1)-p0(1)) - q(2)*(p1(2)-p0(2)); ...
    -p0(2)*(p1(1)-p0(1)) + p0(1)*(p1(2)-p0(2))]; 
b = [p1(1) - p0(1), p1(2) - p0(2);...
    p0(2) - p1(2), p1(1) - p0(1)];
ProjPoint = -(b\a)';
end