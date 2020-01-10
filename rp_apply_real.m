function G = rp_apply_real(X,W,B)
V = W*X;
C = cos(V)
S = sin(V)
G = [C; S];
end