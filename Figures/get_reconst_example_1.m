%env_home = 1;

addpath(genpath('Classes'))
addpath(genpath('Subfuncs'))

prior = 'expertPrior';%prior = 'noPrior';
lik = 'unampLik'; %lik = 'linLik';
%gitsha = 'gitsha_2bd0d720de0995be6b0f1795304839f9877cb6c3';
gitsha = 'gitsha_2bd0d72_evalgit_db4ade8';
trainType = '_rPC_1_origPMgain_useNans';
targetCoverage = '_targetCoverage_10';


dataset_name = 'neurofinder.01.00';
gridType = '_grid_30_7';

%chomp_timestamp = '20190521T173716'; chomp_iter = '1';
%chomp_timestamp = '20190521T204038'; % looong run chomp_iter = '1';
%chomp_timestamp = '20190526T133528'; % with new input image gitsha chomp_iter = '1';
%chomp_timestamp = '20190529T010717'; chomp_iter = '2'; % after recent fixes
chomp_timestamp = '20190529T060454';  chomp_iter = '3'; % after recent fixes % .  iter 2 extra-diagonal, iter 3-diag_cumulant, iter 4 - neither

%chomp_iter = '1';
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

if env_home
  update_visualize_model(model, use_cells,1);
end

%%
[H, W, X, y_orig, y, L] = model.get_fields( 'H', 'W', 'X', 'y_orig','y','L');
szWY = size(model.y);

%use_cells = [3,15,21,57,63,79,103];
%use_cells = [10,30,96,109,165,166,167,168,169]; %chomp_timestamp = '20190526T133528';
use_cells = [67, 79, 125, 126, 127, 128]; % chomp_timestamp = '20190529T010717';
use_cells = [60,68,140,108,109,110,111,140]; % for clustered cells chomp_timestamp = '20190529T060454';  chomp_iter = '2'; % after recent fixes
use_cells = [60,68,140,108,109,110,111,140]; % for single cells chomp_timestamp = '20190529T060454';  chomp_iter = '2'; % after recent fixes
use_cells = [67,79,125:128]; % for single cells chomp_timestamp = '20190529T060454';  chomp_iter = '4'; % after recent fixes

if ~exist('use_cells', 'var')
use_cells = 1:size(model.H,1);
end
use_cells = use_cells(1:min([numel(use_cells),size(H,1)]));

if env_home
  update_visualize_model(model, use_cells,1);
end

H = H(use_cells,:);


cur_area = [min(H(:,1:2),[],1)-((opt.m-1)/2), max(H(:,1:2),[],1)+((opt.m-1)/2)];
cur_area_inds = {cur_area(1):cur_area(3), cur_area(2):cur_area(4)};

%%
if ~env_home

  % Do reconstructions
  [rh,rl,Wfull] = reconstruct_cell( opt, W, X(use_cells,:) );

  full_reconst_diag = cell(opt.mom,1);
  
  % Diagonal reconstructions via rl{}
  for mom1 = 1:opt.mom
    full_reconst_diag{mom1} = zeros(szWY(1:2));
    for c1 = 1:size(rl{mom1},3)
      row_hat = H(c1,1); col_hat = H(c1,2);
      [inds, cut] = mat_boundary(size(full_reconst_diag{mom1}),row_hat-floor(opt.m/2):row_hat+floor(opt.m/2),col_hat-floor(opt.m/2):col_hat+floor(opt.m/2));
      full_reconst_diag{mom1}(inds{1},inds{2}) = full_reconst_diag{mom1}(inds{1},inds{2}) + rl{mom1}(1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2),c1);
    end
  end

  % Get the full covariance of the area and the reconstructions
  inp = load(get_path(opt));
  data = inp.inp.data;
  data_patch = data.proc_stack.Y(cur_area_inds{1},cur_area_inds{2},:);
  szPatch = size(data_patch);
  data_patch_lin = reshape(data_patch, prod(szPatch(1:2)),szPatch(3));
  data_mean = mean(data_patch_lin,2);
  data_cov = (data_patch_lin-data_mean)*((data_patch_lin-data_mean)')./(szPatch(3)-1);
  
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

  
  % Save all results to be loaded for visualisation
  save(['./Examples/reconst_example_' chomp_timestamp '_iter_' chomp_iter '.mat'], ...
    'full_reconst_diag', 'full_reconst_covmat', ...
    'data_mean', 'data_cov', 'szPatch');
else % if on laptop, just load the saved data
  if ~exist(['~/Data/CHOMP_Examples/reconst_example_' chomp_timestamp '_iter_' chomp_iter '.mat'], 'file')
    copyfile(['./Examples/reconst_example_' chomp_timestamp '_iter_' chomp_iter '.mat'], ...
      ['~/Data/CHOMP_Examples/reconst_example_' chomp_timestamp '_iter_' chomp_iter '.mat']);
  end
  load(['~/Data/CHOMP_Examples/reconst_example_' chomp_timestamp '_iter_' chomp_iter '.mat'])
end


%%

if env_home

  clim_moms = {};
  clim_moms{1} = [0, 1e4];
  clim_moms{2} = [0,2.5e7];
  
 my_colormaps = load('Examples/matlab_colormaps.mat');
  cur_colormap = my_colormaps.felfire;
  set(0, 'DefaultFigureColormap',  my_colormaps.felfire)

  
  inds = cur_area_inds;
  for mom1 = 1:opt.mom
  
    mom_mode = 0; %mom_mode = get_hist_mode(opt.cumulant_pixelwise{mom1},500);
    clim = [min(min(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})-mom_mode)), max(max(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})-mom_mode))];
    %clim(1) = 0.
    
    % Show plots of various diagonal reconstructions
    
   
    figure(100+mom1*10+1); imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})-mom_mode ,clim); axis image; colorbar

    figure(100+mom1*10+2); imagesc(full_reconst_diag{mom1}(inds{1},inds{2}),clim); axis image; colorbar

    figure(100+mom1*10+3); imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})-mom_mode-(full_reconst_diag{mom1}(inds{1},inds{2})),clim); axis image; colorbar
  
  end
  
  
   % Show full covariance reconstruction as well
   mom_mode = 0; % mom_mode = get_hist_mode(opt.cumulant_pixelwise{mom1},500);
  clim = [0,6e6];
  figure(201); imagesc(data_cov - mom_mode*eye(size(data_cov,1)),clim); axis image; colorbar
  figure(202); imagesc(full_reconst_covmat,clim); axis image; colorbar
  figure(203); imagesc(data_cov - mom_mode*eye(size(data_cov,1)) - full_reconst_covmat,clim); axis image; colorbar


end


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
