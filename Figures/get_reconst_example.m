env_home = 1;

addpath(genpath('Classes'))
addpath(genpath('Subfuncs'))

prior = 'expertPrior';%prior = 'noPrior';
lik = 'unampLik'; %lik = 'linLik';
gitsha = 'gitsha_2bd0d720de0995be6b0f1795304839f9877cb6c3';
trainType = '_rPC_1_origPMgain_useNans';
targetCoverage = '_targetCoverage_10';


dataset_name = 'neurofinder.01.00';
gridType = '_grid_30_7';

chomp_timestamp = '20190521T173716';

chomp_iter = '1';
use_cells = 1:400;

stamp = ['_preproc2P_' prior '_' lik '_' gitsha trainType targetCoverage gridType];

if env_home
  cur_model = load(['/mnt/gatsby/nfs/data/gergo/Neurofinder_update/' ...
  dataset_name '/preproc2P/CHOMP/output/' dataset_name ...
  stamp '_' ...
  chomp_timestamp '_iter_' chomp_iter '.mat']);
  
  cur_model.model.opt.root_folder = '/mnt/gatsby';
else

  cur_model = load(['/nfs/data/gergo/Neurofinder_update/' ...
    dataset_name '/preproc2P/CHOMP/output/' dataset_name ...
    stamp '_' ...
    chomp_timestamp '_iter_' chomp_iter '.mat']);
  
end

model = cur_model.model;
opt = model.opt;



[H, W, X, y_orig, y, L] = model.get_fields( 'H', 'W', 'X', 'y_orig','y','L');
szWY = size(model.y);

use_cells = [2,11,65,74];

if ~exist('use_cells', 'var')
  use_cells = 1:size(model.H,1);
end
use_cells = use_cells(1:min([numel(use_cells),size(H,1)]));

update_visualize_model(model, use_cells,1);

[rh,rl,Wfull] = reconstruct_cell( opt, W, X(use_cells,:) );

H = H(use_cells,:);

cur_area = [min(H(:,1:2),[],1)-((opt.m-1)/2), max(H(:,1:2),[],1)+((opt.m-1)/2)];
cur_area_inds = {cur_area(1):cur_area(3), cur_area(2):cur_area(4)};

mom1 = 2;
full_reconst = zeros(szWY(1:2));
for c1 = 1:size(rl{mom1},3)
  row_hat = H(c1,1); col_hat = H(c1,2);
  [inds, cut] = mat_boundary(size(full_reconst),row_hat-floor(opt.m/2):row_hat+floor(opt.m/2),col_hat-floor(opt.m/2):col_hat+floor(opt.m/2));
  full_reconst(inds{1},inds{2}) = full_reconst(inds{1},inds{2}) + rl{mom1}(1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2),c1);
end


clim = [0,8e6];
inds = cur_area_inds;
figure; imagesc(full_reconst(inds{1},inds{2}),clim); axis image; colorbar

figure; imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2}),clim); axis image; colorbar

figure; imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})-full_reconst(inds{1},inds{2}),clim); axis image; colorbar

if ~env_home
  inp = load(get_path(opt));
  data = inp.inp.data;
  data_patch = data.proc_stack.Y(inds{1},inds{2},:);
  szPatch = size(data_patch);
  data_patch_lin = reshape(data_patch, prod(szPatch(1:2)),szPatch(3));
  data_mean = mean(data_patch_lin,2);
  data_cov = (data_patch_lin-data_mean)*((data_patch_lin-data_mean)')./(szPatch(3)-1);
  save(['./Examples/reconst_example_' chomp_timestamp '.mat'], 'data_mean', 'data_cov', 'szPatch');
else
  load(['./Examples/reconst_example_' chomp_timestamp '.mat'])
end


% Get covariance reconstruction
mom1 = 2;
full_reconst_covmat = zeros(szPatch(1), szPatch(2), szPatch(1), szPatch(2));
for c1 = 1:size(H,1)
  row_hat = H(c1,1)-cur_area(1); col_hat = H(c1,2)-cur_area(2);
  [inds, cut] = mat_boundary(size(ones(szPatch(1), szPatch(2))),row_hat-floor(opt.m/2):row_hat+floor(opt.m/2),col_hat-floor(opt.m/2):col_hat+floor(opt.m/2));
  full_reconst_covmat(inds{1},inds{2},inds{1},inds{2}) = full_reconst_covmat(inds{1},inds{2},inds{1},inds{2}) ...
    + rh{mom1}(1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2),1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2),c1);
end


full_reconst_covmat = reshape(full_reconst_covmat, prod(szPatch(1:2)), prod(szPatch(1:2)));

% Show full covariance reconstruction as well
figure; imagesc(data_cov,clim); axis image; colorbar
figure; imagesc(full_reconst_covmat,clim); axis image; colorbar
figure; imagesc(data_cov - full_reconst_covmat,clim); axis image; colorbar



%%
% [ROI_mask, ROIs] = getROIs(opt, use_cells);
% 
% 
% % Get neurofinder results
% try
% orig_path = fileparts(fileparts(fileparts(fileparts(get_path(opt)))));
% orig_regions = loadjson([orig_path filesep 'regions' filesep 'regions.json']);
% orig_ROIs = cellfun(@(x)x.coordinates+1, orig_regions, 'UniformOutput', false);
% 
% 
% % Get the centers for the training ROIs
% orig_ROI_centers = cellfun(@(X)round(mean(X*opt.spatial_scale))', orig_ROIs, 'UniformOutput', false);
% orig_H = [cell2mat(orig_ROI_centers)' ones(length(orig_ROI_centers),1)];
% orig_ROI_mask = zeros([size(ROI_mask,1), size(ROI_mask,2), 3]);
% for i1 = 1:size(orig_ROIs,2)
%   cur_color = 0.5*rand(1,3);
%   try
%     for j1 = 1:size(orig_ROIs{i1},1)
%       orig_ROI_mask(orig_ROIs{i1}(j1,1), orig_ROIs{i1}(j1,2),:) = cur_color;
%     end
%   end
% end
% 
% catch
%   orig_H = H(1,:);
%   orig_ROI_mask = zeros([size(ROI_mask,1), size(ROI_mask,2), 3]);
% end
% 
% 
% figure(42); clf; imagesc(opt.cumulant_pixelwise{mom1}); axis image; colorbar;
% hold on; plot(orig_H(:,2), orig_H(:,1), 'or' , 'Linewidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'r', 'MarkerEdgeColor','r');
% 
% figure(7); imagesc(ROI_mask); axis image; colorbar;
% figure(8); imagesc(orig_ROI_mask); axis image; colorbar;
