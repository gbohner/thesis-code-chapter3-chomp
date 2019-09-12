function [Lest, nx1, x1, lasso_groups, S] = matrix_decomp_localise( y, B, I, num_obj)
%MATRIX_DECOMP_LOCALISE Summary of this function goes here
%   Provides locations

% Create the spatial design matrix
m = size(B,1);
num_ss = size(B,ndims(B));
T = size(y,2);
S = zeros(prod(I)+2*m-1, num_ss*(prod(I)+2*num_ss-1));
for i1 = 1:(prod(I)+m-1)
  for n1 = 1:num_ss
    S(i1:(i1+m-1),(i1-1)*num_ss+n1) = B(:,n1);
  end
end
S = S((m+1):(end-m+1),(m+2):(end-m+1+num_ss));
%disp(['spatial design matrix size: ' num2str(size(S))])

lasso_groups = num_ss*ones(round(size(S,2)/num_ss), 1);

%x1 = group_lasso(S, y, 1e-2, lasso_groups, 1e-3, 1.0);
x1 = group_lasso(S, y, 1e0, lasso_groups, 1e-3, 1.0);

x1 = reshape(x1', T*num_ss, prod(I));

nx1 = sqrt(sum(x1.^2,1));
nx1 = nx1(:);

[val, Lest] = sort(nx1, 'descend');

Lest = Lest(1:num_obj);

end

