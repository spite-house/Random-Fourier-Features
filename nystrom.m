% Use Nystrom's method to compute s a matrix G so that G'G is a rank d
% approximation to the kernel matrix
% 
% Nystrom works like this. Partition K as
% K = [K11 K12
%      K21 K22]
% and approximate K22 by returning G so that
% G'G = [K11        K12
%        K21 K21*pinv(K11)*K12]
%
% This approximation is exact if K=[V CV]'[V CV] for some V and C. 
% This holds for example, when the features of some of the points form
% a basis for the features of other points. So this is a fairly restrictive
% form of low-rank for K.
%
function [G,W] = nystrom(X,d,kernel)
N = size(X,2);

% Choose d random points. These form K11
ra = randperm(N);
%ra = 1:d;
p = logical(zeros(1,N));
p(ra(1:d)) = 1;

K11 = kernel(X(:,p));
K12 = kernel(X(:,p),X(:,~p));

[U,S] = svd(K11,0);
Ssqrt = sqrt(diag(S));
Sinvsqrt = Ssqrt; Sinvsqrt(Ssqrt>eps) = 1./Ssqrt(Ssqrt>eps);

T = diag(Sinvsqrt)*U';
G = zeros(d,N);
G(:,p) = diag(Ssqrt)*U';
G(:,~p) = T*K12;

W.T = T;
w.p = p;