% Sample from the fourier bases of the a kernel to obtain a rank d
% approximation for the kernel.
% G is a d by N complex matrix so that G'G approximates K.
function [G,W] = rp_factorize(X,d,kernel)
D = size(X,1);
W = rp_projections(D,d,kernel);
G = rp_apply(X,W);
end
