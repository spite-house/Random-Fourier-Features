% Solve for c in
% (G'G+lambda*I) c = y
% 
% Rather than solving for c, solves for u=Gc.
%
% GG'Gc+lambda*Gc = Gy
% u = (GG'+lambda*I)\Gy
%
% This function is different from lowranksolver2 in that it
% takes GG' and Gy as arguments.
function u = lowranksolver3(GG,Gy,lambda)
I_d = eye(size(GG,1));
%u = (I_d*double(lambda)+GG)\Gy;
u = symmlq(@(v)(double(lambda)*v(:) + GG'*v(:)),Gy,1e-6,2000);
end
