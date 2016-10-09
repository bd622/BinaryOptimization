clear; 
close all;

addpath ./utils/
addpath ./eval/

feature = 'RBF';
dataset = 'cifar10_gist';

run('pre_data.m');

% parameters
opt.maxItr = 1;
opt.lambda = 0.25;
opt.delta = 1e3;
opt.rho = 1e2;

n = size(feaTrain,1);
bits_set = [16 32 64 96 128];

Pre = zeros(1,length(bits_set));
Rec = zeros(1,length(bits_set));
MAP = zeros(1,length(bits_set));
PreTopK = zeros(1,length(bits_set));
acc = zeros(1,length(bits_set));

for ii = 1:length(bits_set)
    nbits = bits_set(ii);

    [~,F,H] = DPLM_L2(feaTrain,traingnd,nbits,opt);

    display('Evaluation...');
    tH = sign(feaTest*F.W);
   
    hammTrainTest = 0.5*(nbits - H*tH');
    % hash lookup: precision and reall
    hammRadius = 2;
 
    Ret = (hammTrainTest <= hammRadius+0.00001);
    [Pre(ii), Rec(ii)] = evaluate_macro(cateTrainTest, Ret);
    
    % hamming ranking: MAP and Pre, recall
    [~, HammingRank]=sort(hammTrainTest,1);
    MAP(ii) = cat_apcal(cateTrainTest,HammingRank)
    
    [PreTopK(ii)] = cat_ap_topK(cateTrainTest,HammingRank, 500)

    W = RRC(double(H),traingnd,1);
    [~,label] = max(double(tH)*W,[],2);
    acc(ii) = sum(testgnd(:)==label)/length(label);
end

