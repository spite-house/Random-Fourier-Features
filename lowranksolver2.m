% Solve for c in
% (G'G+lambda*I) c = y
%
% Rather than returning c, returns u=Gc.
%
% GG'Gc+lambda*Gc = Gy
% u = (GG'+lambda*I)\Gy
function u = lowranksolver2(G,y,lambda)

% THIS IS FOR DEBUGGING ONLY
% if issparse(G)
%   I = speye(size(G,1));
%   u = (I*lambda+G*G')\(G*y);
% else
%   I = eye(size(G,1));
%   u = (I*lambda+G*G')\(G*y);
% end

% tic
% u = pcg(@(v)(lambda*v(:) + G*(G'*v(:))),G*y(:),1e-6,500);
% toc

% THIS IS THE FASTEST ONE
u = symmlq(@(v)(double(lambda)*v(:) + G*(G'*v(:))),G*y(:),1e-6,2000);

% tic
% u = cgs(@(v)(lambda*v(:) + G*(G'*v(:))),G*y(:),1e-6,500);
% toc

% tic
% u = bicgstab(@(v)(lambda*v(:) + G*(G'*v(:))),G*y(:),1e-6,500);
% toc
end
