num_obj = 10;
num_ss = 3;
T = 300;
SNR = 10;

I = [64,64];
bsize = 5;
% %Random basis funcs
% B = reshape(randn(bsize^numel(I),num_ss),[bsize*ones(1,numel(I)),num_ss]); %Gotta make sure to center basis functions, noncentered one can cause constant shift (original code has centering)

%Try nonrandom B
[~, mask] = transform_inds_circ(0,0,150,bsize,2,0);
B(:,:,1) = mask;
[~, mask] = transform_inds_circ(0,0,150,bsize,2,1);
B(:,:,2) = mask;
[~, mask] = transform_inds_circ(0,0,150,bsize,1,0);
B(:,:,3) = mask;

%Normalize basis funcs
for i1 = 1:num_ss
  Bdims = cell(1,ndims(B)); [Bdims{1:end-1}] = deal(':');
  Bdims{end} = i1;
  B(Bdims{:}) = B(Bdims{:}) ./ norm(reshape(B(:,:,i1),1,[]));
end

%X = randn(num_obj,num_ss,T)*SNR; % change in variance
X = reshape(laprnd(num_obj*num_ss,T,0,SNR),num_obj, num_ss, T); % high
%kurtosis
%X = rand(num_obj,num_ss,T)+1; % high skewness
%X = exprnd(SNR,num_obj,num_ss,T); % high skewness

%Test X that has different distributions in different basis directions
%X = exprnd(SNR,num_obj,1,T);
%X(:,end+1:end+num_ss-1,:) = rand(num_obj,num_ss-1,T)+1*SNR;

L = floor(rand(num_obj,numel(I)) * 44 + 13); %for 64 x 64 images

y = create_data(I,T,L,X,B,1);

figure(1234); imagesc(mean(y,ndims(y)))

%%

%Get moment estimators
mom = 4;
M = get_moments( y, B, mom, 'diagonal' ); %moments by data points


%% Run matching pursuit
O = matching_pursuit( M, B, I, num_obj);
[O.moments]
[sortrows([O.loc]') zeros(num_obj,1) sortrows(L)]

%{
%%

%Find eigenvectors of moment estimators
[U,S] = svd(M,0);

%%
values = U(:,1)'*M;
%proj on first eigenvec of data points
values = (values(1,:));
if skewness(values)>=0
    [~, inds] = sort(values,'descend');
else
    [~, inds] = sort(values,'ascend');
end
locs = 1:prod(I);
locs_sorted = locs(inds);
L_estimate = cell(numel(I),1);
[L_estimate{:}] = ind2sub(I,locs_sorted(1:num_obj));
[sortrows(cell2mat(L_estimate)'), zeros(num_obj,2), sortrows(L)]
[U(:,1)'] %moment estimator weightings
%%
%mean image
ys = mean(mean(y,4),3);
figure(1); imagesc(ys')
colorbar;

%estimator projection onto first eigenvec image
figure(5); 
imagesc(mean(reshape(values,I),3)');
colorbar;



%Get locations from projections onto the first eigenvector


%%
ys = mean(mean(y,4),3);
figure(1); imagesc(ys')
colorbar;
figure(2); %imagesc(mean(std(y,1,ndims(y)),3)');
imagesc(mean(sum(y.^2,ndims(y)),3)')
colorbar

figure(3); %imagesc(mean(skewness(y,1,ndims(y)),3)');
imagesc(mean(sum(y.^3,ndims(y)),3)')
colorbar

figure(4); %imagesc(mean(kurtosis(y,1,ndims(y)),3)');
imagesc(mean(sum(y.^4,ndims(y)),3)')
colorbar

%}