% Solve for c in
% (G'G+lambda*I) c = y
%
% Uses Woodbury formula to compute
% (lambda*I+G'G)^-1 = I/lambda - I/lambda^2 * G'*(I+G*G'/lambda)^-1 G
function c = lowranksolver(G,y,lambda)
I_d = speye(size(G,1));
c = y/lambda - G'*((I_d+G*G'/lambda)\(G*y))/lambda^2;
end
