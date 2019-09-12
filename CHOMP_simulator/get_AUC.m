function AUC = get_AUC( L, Lest )
%GET_AUC Summary of this function goes here
%   Detailed explanation goes here

ROC = zeros(1,numel(Lest));
ROC_count = zeros(1,numel(Lest));

for i1 = 1:numel(Lest)
  [idx, dl] = knnsearch(Lest(1:i1,1),L);
  ROC(i1) = sum(dl==0)/i1;
  ROC_count(i1) = i1./numel(Lest);
end

AUC = sum((ROC-ROC_count).*((ROC-ROC_count)>0))./sum(ones(1,numel(Lest))-ROC_count);

if AUC>1
  disp('hmmm');
end