clear; 
close all;

addpath ./utils/
addpath ./eval/
addpath ./testbed/

feature = 'raw';
dataset = 'cifar10_gist';

run('pre_data.m');

% parameters
opt.maxItr = 10;
opt.lambda = 2.2;
opt.delta = 3e3;
opt.rho = 1e2;

% We adopt the anchor graph to compute laplacian matrix
load anchor_1000_cifar
A = AffinityMatrix(feaTrain, anchor, 3, 0);
L = diag(sum(A,2)) - A;
clear A;

bits_set = [64 128];

Pre = zeros(1, length(bits_set));
Rec = zeros(1, length(bits_set));
MAP = zeros(1, length(bits_set));
PreTopK = zeros(1, length(bits_set));
acc = zeros(1, length(bits_set));

for ii = 1:length(bits_set)
    bits = bits_set(ii);
    
    [F, H] = DPLM_SH(feaTrain, L, bits, opt);
    
    display('Evaluation...');
    H = sign(feaTrain*F);
    tH = sign(feaTest*F);

    hammTrainTest = 0.5*(bits - H*tH');
    % hash lookup: precision and reall
    hammRadius = 2;
    
    Ret = (hammTrainTest <= hammRadius+0.00001);
    [Pre(ii), Rec(ii)] = evaluate_macro(cateTrainTest, Ret);
    
    % hamming ranking: MAP and Pre, recall
    [~, HammingRank]=sort(hammTrainTest,1);
    MAP(ii) = cat_apcal(cateTrainTest, HammingRank)
    [PreTopK(ii)] = cat_ap_topK(cateTrainTest,HammingRank, 500)
    
    W = RRC(double(H),traingnd,1);
    [~,label] = max(double(tH)*W,[],2);
    acc(ii) = sum(testgnd(:)==label)/length(label)
end
