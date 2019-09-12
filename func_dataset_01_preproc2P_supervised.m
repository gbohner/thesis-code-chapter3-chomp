function func_dataset_01_preproc2P_supervised(preproc2P_stamp)

% preproc_stamp is the stamp within the "images_" folder

%close all;

%Example local run

cd(fileparts(mfilename('fullpath')));
addpath(genpath('.'))

setenv('CHOMP_ROOT_FOLDER', '');

opt = chomp_options(); %Initialize default options

% Data paths and types
opt.data_path = [...
  '/nfs/data/gergo/Neurofinder_update/neurofinder.01.00' ...
  '/preproc2P/images' preproc2P_stamp ...
  '/image00001.tif'];
opt.data_type = 'frames';
opt.src_string = '.tif';

% Set the subfolders names - neurofinder specific
opt.input_folder = [fileparts(fileparts(opt.data_path)) '/CHOMP/input/'];
opt.output_folder = [fileparts(fileparts(opt.data_path)) '/CHOMP/output/'];
opt.results_folder = [fileparts(fileparts(opt.data_path)) '/CHOMP/results/'];


% CHOMP preprocessing options
cell_max_size = 27;
opt.spatial_scale = 0.7;%1.0;%1.2;
opt.time_scale = 0.4;
opt.whiten = 1;
opt.stabilize = 0;
opt.smooth_filter_sigma = 0.4000;
opt.smooth_filter_mean = 1.4000;
opt.smooth_filter_var = 4;

% CHOMP model options
opt.NSS = 1;
opt.KS = 2;
opt.mom = 1;
%opt.mom_weights = [0.2,1]; %[0,1,1]; %[0, 0, 0, 1];
opt.init_model = 'donut_conv'; %{'donut_conv', 'pointlike'};%{'donut_four', 'pointlike'}; %'donut_two'; %'donut_conv'; %'pointlike'; %'filled'; %{'filled', 'donut'};
opt.niter = 3; %opt.learn =0; %opt.niter =5;
opt.learn_decomp = 'COV_RAW';
opt.diag_tensors = 0; %1;
opt.ROI_params = [0.7];
opt.W_weight_type = 'decomp'; %'decomp'; % 'uniform' / 'decomp' - latter using SVD weights to penalise use of bases (but only works with diag_tensors=1 for now)
opt.ROI_type = 'ones_origsize';%'quantile_dynamic_origsize';
opt.ROI_params = 0.6;

% Mask "bad" areas
opt.mask = 0;

% CHOMP derived options
opt.m = round(cell_max_size*opt.spatial_scale) + (1 - mod(round(cell_max_size*opt.spatial_scale),2));

[~, tmp1, tmp2] = fileparts(fileparts(fileparts(fileparts(opt.data_path)))); % Important if you want to have nice file prefixes (corresponding to main folder name)
opt.file_prefix = [tmp1 tmp2 '_preproc2P' preproc2P_stamp];

% Visualisation and verbosity options
opt.fig = 0;
opt.verbose = 3;

%%

%Learning with 150 cells;
opt.cells_per_image = 150;
opt.spatial_push = @(grid_dist)1.0*(grid_dist>=(2*opt.m)); % For learning, do not allow overlapping regions
%opt.spatial_push = [];
[opt, ROI_mask, ROIs] = chomp(opt);

% % %% Supervised learning
% % 
% % orig_regions = py.neurofinder.load([fileparts(fileparts(fileparts(get_path(opt)))) filesep 'regions' filesep 'regions.json']);
% % orig_ROIs = cellfun(@matpy.nparray2mat, cell(orig_regions.coordinates), 'UniformOutput', false);
% % 
% % ROI_centers = cellfun(@(X)round(mean((X-1)*opt.spatial_scale+1))', orig_ROIs, 'UniformOutput', false);
% % true_H = [cell2mat(ROI_centers)' ones(length(ROI_centers),1)];
% % 
% % for type = 1:inp.opt.NSS
% %   [W(:,inp.opt.Wblocks{type}), Sv] = update_dict(inp.data,true_H,model.W(:,inp.opt.Wblocks{type}),inp.opt,n+2,type);
% %   if strcmp(opt.W_weight_type, 'decomp')
% %     inp.opt.W_weights(inp.opt.Wblocks{type}) = Sv;
% %   end
% % end
% % 
% % if inp.opt.fig >0
% %   update_visualize(model.y,true_H, ...
% %     reshape(W,model.opt.m,model.opt.m,size(model.W,ndims(model.W))),...
% %     model.opt,1);
% % end

 %
%Inferring with 1200 cells given the supervised learned subspace
opt.cells_per_image = 1200;
opt.init_iter = opt.niter; %Just running the inference
opt.niter = opt.niter+1; %Just running the inference
opt.spatial_push = @(grid_dist)1.0*(grid_dist>=(opt.m/3));
%opt.spatial_push = @(grid_dist)logsig(0.7*grid_dist-floor(opt.m-1)).*(grid_dist>=(2*opt.m/3)); %@(grid_dist, sharp)logsig(sharp*grid_dist-floor(sharp*2*obj.m/2-1));
% opt.spatial_push = [];
%opt.spatial_push = @(grid_dist)1.0*(grid_dist>=(opt.m/3)); % For inference, be less restrictive on spatial push
%opt.spatial_push = @(grid_dist)logsig(0.5*grid_dist-floor(opt.m/2-1)); %@(grid_dist, sharp)logsig(sharp*grid_dist-floor(sharp*2*obj.m/2-1));
[opt, ROI_mask, ROIs] = chomp(opt);

get_neurofinder_results( opt, ROIs, true )


%get_cell_timeseries(opt);
% 
% Learn the classifier and ROI regressors given the output of CHOMP and training data
trainedModel = train_neurofinder_ROIs( opt, true );


%% Run the inference on a new dataset

% 
% % Set the new data paths
% opt.data_path = '/nfs/data/gergo/Neurofinder_update/neurofinder.00.01/images/image00001.tiff';
% % Set the subfolders names - neurofinder specific
% opt.input_folder = [fileparts(fileparts(opt.data_path)) '/CHOMP/input/'];
% opt.output_folder = [fileparts(fileparts(opt.data_path)) '/CHOMP/output/'];
% opt.results_folder = [fileparts(fileparts(opt.data_path)) '/CHOMP/results/'];
% 
% [~, tmp1, tmp2] = fileparts(fileparts(fileparts(opt.data_path))); % Important if you want to have nice file prefixes (corresponding to main folder name)
% opt.file_prefix = [tmp1 tmp2];
% 
% 
% %opt.timestamp = [];
% opt.init_iter = 0;
% opt.niter = 1;
% opt.learn = 0;
% 
% % Run the inference on the new dataset
% [opt, ROI_mask, ROIs] = chomp(opt);
% 
% % Get the neurofinder-trained ROIs given the CHOMP results
% [ROI_mask, ROIs] = get_neurofinder_ROIs( opt, trainedModel, true );
% 
% %% Show evaluation results with neurofinder script
% 
% 
% get_neurofinder_results( opt, ROIs )
% 
% 


%% Run it also on the test dataset

% Load the W from the learned model
load(get_path(opt, 'output_iter', opt.niter), 'model');
opt.init_W = model.W;
opt.init_model = 'given';
opt.W_weights = model.opt.W_weights;

% Set no learning just inference
opt.init_iter = 0;
opt.niter = 1;
opt.learn = 0;
opt.cells_per_image = 1200;

% Set the new data paths
opt.data_path = [...
  '/nfs/data/gergo/Neurofinder_update/neurofinder.01.00.test' ...
  '/preproc2P/images' preproc2P_stamp ...
  '/image00001.tif'];
% Set the subfolders names - neurofinder specific
opt.input_folder = [fileparts(fileparts(opt.data_path)) '/CHOMP/input/'];
opt.output_folder = [fileparts(fileparts(opt.data_path)) '/CHOMP/output/'];
opt.results_folder = [fileparts(fileparts(opt.data_path)) '/CHOMP/results/'];

[~, tmp1, tmp2] = fileparts(fileparts(fileparts(fileparts(opt.data_path)))); % Important if you want to have nice file prefixes (corresponding to main folder name)
opt.file_prefix = [tmp1 tmp2 '_preproc2P' preproc2P_stamp];


[opt, ROI_mask, ROIs] = chomp(opt);
[ROI_mask, ROIs] = get_neurofinder_ROIs( opt, trainedModel );

ROI_to_json(opt, ROIs);


%% Run it also on the test dataset 2
% Set the new data paths
opt.data_path = [...
  '/nfs/data/gergo/Neurofinder_update/neurofinder.01.01.test' ...
  '/preproc2P/images' preproc2P_stamp ...
  '/image00001.tif'];
% Set the subfolders names - neurofinder specific
opt.input_folder = [fileparts(fileparts(opt.data_path)) '/CHOMP/input/'];
opt.output_folder = [fileparts(fileparts(opt.data_path)) '/CHOMP/output/'];
opt.results_folder = [fileparts(fileparts(opt.data_path)) '/CHOMP/results/'];

[~, tmp1, tmp2] = fileparts(fileparts(fileparts(fileparts(opt.data_path)))); % Important if you want to have nice file prefixes (corresponding to main folder name)
opt.file_prefix = [tmp1 tmp2 '_preproc2P' preproc2P_stamp];


[opt, ROI_mask, ROIs] = chomp(opt);
[ROI_mask, ROIs] = get_neurofinder_ROIs( opt, trainedModel );

ROI_to_json(opt, ROIs);