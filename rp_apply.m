function G = rp_apply(X,W)
G = exp(i*W*X);
%G = [cos(W*X); sin(W*X)];
end
