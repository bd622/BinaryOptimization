function [F, B] = DPLM_SH(X, L, bits, opt)
% optimize the graph hashing problem by DPLM.
% B : learned binary codes.
% F : learned Hash function.

tol = 1e-6;
nu = (1/opt.lambda);

n = size(L, 1);
B = sign(randn(n, bits));

for ii = 1:opt.maxItr
    B0 = B;
    a = 2*L*B;
    b = (opt.delta/(n^2))*B*(B'*B);
    c = (opt.rho/(n^2))*repmat(sum(B),n,1);
    grad = a + b + c;
    B = sign(B - grad*nu);
    
    conv_B = norm(B-B0,'fro')
    if conv_B < tol*norm(B0,'fro')
       break;
    end
end

[F, ~, ~] = RRC(X, B, 1);


