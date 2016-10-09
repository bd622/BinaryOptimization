function [ap] = cat_apcal(cateTrainTest, IX)
% ap=apcal(score,label)
% average precision (AP) calculation 

[numtrain, numtest] = size(IX);

apall = zeros(1,numtest);
num_return_NN = numtrain; % only compute MAP on returned top 5000 neighbours.
    
for i = 1 : numtest
    y = IX(:,i);
    x=0;
    p=0;
    
    for j=1:num_return_NN
        if cateTrainTest(y(j),i)==1
            x=x+1;
            p=p+x/j;
        end
    end  
    
    if p==0
        apall(i)=0;
    else
        apall(i)=p/x;
    end   
end

ap = mean(apall);
