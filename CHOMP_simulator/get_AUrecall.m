function AUC = get_AUrecall( L, Lest, I )
%GET_AUC Summary of this function goes here
%   Detailed explanation goes here

ROC = zeros(1,numel(Lest));
ROC_chance = zeros(1,numel(Lest));

for i1 = 1:numel(Lest)
  [idx, dl] = knnsearch(Lest(1:i1,1),L);
  ROC(i1) = sum(dl==0)/i1;
  ROC_chance(i1) = get_chance(i1, prod(I))/i1;
end

AUC = sum((ROC-ROC_chance).*((ROC-ROC_chance)>0))./sum(ones(1,numel(Lest))-ROC_chance); %Normalized by perfect recall

if AUC>1
  disp('hmmm');
end

function pr = get_chance(s, numI) %Expected value of picking s random locations out of I
  pr = 0;
  warning('off','MATLAB:nchoosek:LargeCoefficient')
  for sprime = 1:s
    pr = pr + sprime * nchoosek(s, sprime) * nchoosek(numI-s, s-sprime);
  end  
  pr = double(pr * 1./nchoosek(numI,s));
  warning('on','MATLAB:nchoosek:LargeCoefficient')
end

end