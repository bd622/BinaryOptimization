function [G,F,B] = DPLM_L2(X,y,nbits,opt)
% optimize the supervised hashing problem by DPLM.
% B : learned binary codes.
% F : learned Hash function.
% G : learned classification matrix.

tol = 1e-6;
max_iter_B= 10;
nu = (1/opt.lambda);

n = size(X,1);

% label matrix N x c
if isvector(y)
    Y = sparse(1:length(y), y, 1); Y = full(Y);
else
    Y = y;
end
Y = nbits*Y;

G = [];

% init B
randn('seed',0);
B = sign(randn(n,nbits));
Wg0 = zeros(nbits,size(Y,2));
i = 0;

while i < opt.maxItr
    i=i+1;
    
    fprintf('Iteration  %03d: \n',i)
    % G-step
    Wg = RRC(B, Y, 1);
    G.W = Wg;
    
    conv_w = norm(Wg-Wg0,'fro');
    if conv_w < tol*norm(Wg0,'fro')
        break
    end
    Wg0 = Wg;
    
    % B-step
    for ix_B = 1:max_iter_B
        B0 = B;
        a = (B*Wg - Y)*Wg';
        b = (opt.delta/(n^2))*B*(B'*B);
        c = (opt.rho/(n^2))*repmat(sum(B),n,1);
        grad_p = a + b + c;
        
        B = sign(B - nu*grad_p);
        
        conv_B = norm(B-B0,'fro')
        if conv_B < tol*norm(B0,'fro')
           break
        end

    end    

   fprintf('\n');
end

% F-step
[WF, ~, ~] = RRC(X, B, 1);
F.W = WF;

