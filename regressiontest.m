function [perf, error, time] = regressiontest(Xtrain,ytrain,Xtest,ytest,kernel,method,lambda,varargin)


N = size(Xtrain,2);

kernels = struct('gaussian',@kernel_gaussian,...
                 'laplacian',@kernel_laplacian,...
                 'linear',@kernel_linear);

ytrain_mean = mean(ytrain);
ytrain = ytrain-ytrain_mean;

perf.method = method;
perf.lambda = lambda;
%perf.kernels = kernels;
perf.kernel = kernel;
perf.ytrain_mean = ytrain_mean;

fprintf('Factoring...');
tic
switch method
 case 'calibvar'
  perf = calibvar(Xtrain); return;
 case 'exact'
  K = feval(getfield(kernels,kernel),Xtrain);
 case 'nystrom'
  d = varargin{1};
  [G,W] = nystrom(Xtrain,d,getfield(kernels,kernel));
 case 'rp_factorize'
  d = varargin{1};
  [G,W] = rp_factorize(Xtrain,d,kernel);
 case 'rp_factorize_large'
  d = varargin{1};
  [GG,Gy,W] = rp_factorize_large(Xtrain,ytrain,d,kernel, 500);
 case 'rp_factorize_large_real'
  d = varargin{1};
  [GG,Gy,W,B] = rp_factorize_large_real(Xtrain,ytrain,d,kernel,1000);
 case 'rpbin'
  d = varargin{1};
  [G,W] = rpbin(Xtrain,d,kernel);
  %spy(G);drawnow
 otherwise
  error('Don''t have a test like that');
end
perf.factorizetime = toc;
time = perf.factorizetime;


if exist('d','var')
  perf.rps = d;
end
if exist('W','var')
  perf.W = W;
end
if exist('B', 'var')
    perf.B = B;
end


fprintf('solving...');
tic;
switch method
 case 'exact'
  c = (K+eye(N)*lambda)\ytrain(:);
  perf.c = c;
 case 'nystrom'
  c = lowranksolver(G,ytrain(:),lambda);
  perf.c = c;
 case {'rp_factorize','rpbin'}
  u = lowranksolver2(G,ytrain(:),lambda);
  perf.u = u;
 case {'rp_factorize_large','rp_factorize_large_real'}
  u = lowranksolver3(GG,Gy,lambda);
  perf.u = u;
end
perf.solvetime = toc;

if ~isempty(Xtest)
  perf = evalregression(perf,Xtest,ytest);
  error = perf.error
end
end


function v = calibvar(X)
p = logical(binornd(1,2000/size(X,2),1,size(X,2)));
R= L2_distance(X(:,p),X(:,p)).^2;
v= mean(R(:));
end


function K = kernel_linear(X,Y)
if nargin==1
    Y = X;
end
K = X'*Y;
end

function K = kernel_gaussian(X,Y)
if nargin==1
    Y = X;
end
R = L2_distance(X,Y);
K = exp(-R.^2);
end

function K = kernel_laplacian(varargin)
R = L1_distance(varargin{:});
K = exp(-R);
end
