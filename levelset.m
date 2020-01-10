function levelset
load polygons;

I = I(1:4:end,1:4:end);
figure(1); imagesc(I);

[i,j] = find(I);
N = length(i);
Xtrain = [i(:),j(:)]';
ytrain = ones(1,N);

[i,j] = meshgrid(1:1:size(I,2), 1:2:size(I,1));
Xtest = [i(:), j(:)]';


s = 0.3;
lambda = 0.01;

ytest  = rp(s*Xtrain,ytrain,s*Xtest,400,lambda);
figure(2); draw(i,j,ytest);

ytest  = exact(s*Xtrain,ytrain,s*Xtest,lambda);
figure(3); draw(i,j,ytest);

end

function draw(i,j,ytest)
ytest  = reshape(ytest,size(i,1),size(i,2));
surf(i,j,ytest);
axis vis3d
end

function ytest  = rp(Xtrain,ytrain,Xtest,d,lambda)
[G,W] = rp_factorize(Xtrain,d,'gaussian');
u = lowranksolver2(G,ytrain(:),d*lambda);
fprintf('Evaluating...');
tic
Gtest = rp_apply(Xtest,W);
ytest = real(u'*Gtest);
toc
end

function ytest  = exact(Xtrain,ytrain,Xtest,lambda)
N = size(Xtrain,2);
K = exp(-L2_distance(Xtrain,Xtrain));
c = (K+eye(N)*lambda)\ytrain(:);

fprintf('Evaluating...');
tic
Ktest  =exp(-L2_distance(Xtest,Xtrain));
ytest  = Ktest*c;
toc
fprintf('done.\n');
end
