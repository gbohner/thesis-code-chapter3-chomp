%env_home = 0;

addpath(genpath('Classes'))
addpath(genpath('Subfuncs'))

prior = 'expertPrior';%prior = 'noPrior';
lik = 'unampLik'; %lik = 'linLik';
%gitsha = 'gitsha_2bd0d720de0995be6b0f1795304839f9877cb6c3';
gitsha = 'gitsha_2bd0d72_evalgit_db4ade8';
trainType = '_rPC_1_origPMgain_useNans';
targetCoverage = '_targetCoverage_10';
gridType = '_grid_30_7';

% dataset_name = 'neurofinder.00.00';
% chomp_timestamp = '20190521T231845';

dataset_name = 'neurofinder.01.00';
%chomp_timestamp = '20190521T173716'; % 01.00, used for reconstructing
%chomp_timestamp = '20190521T204038'; % looong run
%chomp_timestamp = '20190526T133528';  % new run 
chomp_timestamp = '20190526T190317';

% dataset_name = 'neurofinder.02.00';
% % chomp_timestamp = '20190516T153006'; % This doesn't have
% % opt.cumulant_pixelwise, so doesn't work with updated training
% chomp_timestamp = '20190522T012238';
% 
% dataset_name = 'neurofinder.04.00';
% chomp_timestamp = '20190522T014112';

chomp_iter = '1';
use_cells = 1:330;

stamp = ['_preproc2P_' prior '_' lik '_' gitsha trainType targetCoverage gridType];

if env_home
  data_dir = '/mnt/gatsby/nfs/data/gergo/Neurofinder_update/';
  cur_model = load([data_dir ...
  dataset_name '/preproc2P/CHOMP/output/' dataset_name ...
  stamp '_' ...
  chomp_timestamp '_iter_' chomp_iter '.mat']);
  
  cur_model.model.opt.root_folder = '/mnt/gatsby';
else
  data_dir = '/nfs/data/gergo/Neurofinder_update/';
  cur_model = load([data_dir ...
    dataset_name '/preproc2P/CHOMP/output/' dataset_name ...
    stamp '_' ...
    chomp_timestamp '_iter_' chomp_iter '.mat']);
  
end


model = cur_model.model;
opt = model.opt;



[H, W, X, y_orig, y, L] = model.get_fields( 'H', 'W', 'X', 'y_orig','y','L');
szWY = size(model.y);

%clear use_cells
if ~exist('use_cells', 'var')
  use_cells = 1:size(model.H,1);
end
use_cells = use_cells(1:min([numel(use_cells),size(H,1)]));

update_visualize_model(model, use_cells,0);

%%
% clear trainedModel
% 
% trainedModel = train_neurofinder_reconst_ROIs(opt, 1); % train only on diag_cumulants
% % THIS ONE CRASHES %trainedModel = train_neurofinder_reconst_ROIs(opt, 1, 1); % train also on offdiag cumulants (or if opt.opt.diag_cumulants_offdiagonly, only on offdiag)
% 
% if ~env_home
%   save(['./Examples/trainedModel_' chomp_timestamp '.mat'], 'trainedModel');
% else
%   if ~exist('trainedModel','var')
%     load(['./Examples/trainedModel_' chomp_timestamp '.mat'])
%   else
%     fprintf('trainedModel is already in the Workspace')
%   end
% end


%%
%[ROI_image, ROIs] = get_neurofinder_reconst_ROIs( opt, trainedModel, 0 );

%opt.ROI_type = 'ones_origsize';
%[ROI_image, ROIs] = getROIs( opt );

%figure; imagesc(ROI_image)

%get_neurofinder_results(opt, ROIs, 1);

%%

max_ROIs = 300;

dataset_name_cur = [dataset_name '.test'];

cur_model = load([data_dir ...
dataset_name_cur '/preproc2P/CHOMP/output/' dataset_name_cur ...
stamp '_' ...
chomp_timestamp '_iter_' chomp_iter '.mat']);

if env_home, cur_model.model.opt.root_folder = '/mnt/gatsby'; end

model = cur_model.model;
opt = model.opt;

%opt.ROI_type = 'ones_origsize';
%[ROI_image, ROIs] = getROIs( opt );
[ROI_image, ROIs] = get_neurofinder_reconst_ROIs( opt, trainedModel );
ROI_to_json(opt, ROIs(1:min(max_ROIs,length(ROIs))));

%%
dataset_name_cur = [dataset_name(1:end-1) '1.test'];

cur_model = load([data_dir ...
dataset_name_cur '/preproc2P/CHOMP/output/' dataset_name_cur ...
stamp '_' ...
chomp_timestamp '_iter_' chomp_iter '.mat']);

if env_home, cur_model.model.opt.root_folder = '/mnt/gatsby'; end


model = cur_model.model;
opt = model.opt;

%
%[ROI_image, ROIs] = getROIs( opt );
[ROI_image, ROIs] = get_neurofinder_reconst_ROIs( opt, trainedModel );
ROI_to_json(opt, ROIs(1:min(max_ROIs,length(ROIs))));


%%
% use_cells = 1:330;
% %opt.ROI_type = 'brightest_pixel_origsize';
% opt.ROI_type = 'ones_origsize';
% [ROI_mask, ROIs] = getROIs(opt, use_cells);
% 
% get_neurofinder_results(opt, ROIs);
% %ROI_to_json(opt, ROIs);
% 
% 
% figure; imagesc(ROI_mask)

%[rh,rl,Wfull] = reconstruct_cell( opt, W, X(use_cells,:) );

% update_visualize(full_reconst_mom1,model.H, ...
%   reshape(model.W,model.opt.m,model.opt.m,size(model.W,ndims(model.W))),...
%   model.opt,1,1);


% figure(4); clf; imagesc(opt.cumulant_pixelwise{mom1}); axis image; colorbar;
% hold on; plot(H(:,2), H(:,1), 'or' , 'Linewidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'r', 'MarkerEdgeColor','r');
