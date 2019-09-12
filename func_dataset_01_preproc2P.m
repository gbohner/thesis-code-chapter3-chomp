function [out_timestamp, saved_opts] = func_dataset_01_preproc2P(preproc2P_stamp, input_opts)

% preproc_stamp is the stamp within the "images_" folder
% input_opts contains a struct used to overwrite the default options set
% below

if nargin<2
  input_opts = struct();
end

saved_opts = {};


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
cell_max_size = 23;
opt.spatial_scale = 1.0;%0.375;%1.0;%1.2;
opt.time_scale = 0.1;
opt.denoise_medianfilter = 1;
opt.subtract_background = 1;
opt.whiten = 0;
opt.stabilize = 0;
opt.smooth_filter_sigma = 0.4000;
opt.smooth_filter_mean = 1.4000;
opt.smooth_filter_var = 4;

% CHOMP derived options
opt = struct_merge(opt, input_opts); % add the inputted changes before opt.m is calculated
opt.m = round(cell_max_size*opt.spatial_scale) + (1 - mod(round(cell_max_size*opt.spatial_scale),2));


% CHOMP model options
opt.NSS = 1;
opt.KS = 8;
opt.mom = 2;
opt.mom_weights = [0.3,1.];%[1,1,1,1]; %[0,1,1]; %[0, 0, 0, 1];
opt.init_model = 'neurofinder_preproc2P';%'donut_four';%'neurofinder_preproc2P';%'donut_four';%'donut_conv'; %'neurofinder_preproc2P'; %'donut_four'; %'donut_conv'; %{'donut_conv', 'pointlike'};%{'donut_four', 'pointlike'}; %'donut_two'; %'donut_conv'; %'pointlike'; %'filled'; %{'filled', 'donut'};
opt.niter = 1; opt.learn =0;
opt.learn_decomp = 'COV_RAW';
opt.diag_tensors = 0; %1;
opt.diag_cumulants = 1;
opt.diag_cumulants_offdiagonly = 0;
opt.W_addflat = 0;
opt.local_peaks = 1;
opt.W_force_round = 1;
opt.blank_reconstructed = 1;
opt.W_weight_type = 'uniform'; %'decomp'; % 'uniform' / 'decomp' - latter using SVD weights to penalise use of bases (but only works with diag_tensors=1 for now)
opt.ROI_type = 'ones_origsize';%'quantile_origsize'; %'ones_origsize';%'quantile_dynamic_origsize';
opt.ROI_params = 0.5;

% Mask "bad" areas, edges etc
opt.mask = 1;
opt.mask_image = ones([512,512]);
opt.mask_image(:,1:round(cell_max_size/2.)) = 0;
opt.mask_image(:,(end-round(cell_max_size/2.)):end) = 0;
opt.mask_image(1:round(cell_max_size/2.),:) = 0;
opt.mask_image((end-round(cell_max_size/2.)):end, :) = 0;
opt.mask_image= imresize(opt.mask_image, opt.spatial_scale,'nearest');


[~, tmp1, tmp2] = fileparts(fileparts(fileparts(fileparts(opt.data_path)))); % Important if you want to have nice file prefixes (corresponding to main folder name)
opt.file_prefix = [tmp1 tmp2 '_preproc2P' preproc2P_stamp];

% Visualisation and verbosity options
opt.fig = 0;
opt.verbose = 3;

%%

%Learning with 150 cells;
opt.cells_per_image = 150; % very few cells as we just set this up to learn the initial basis
opt.spatial_push = @(grid_dist)1.0*(grid_dist>=(2*opt.m)); % For learning, do not allow overlapping regions
%opt.spatial_push = @(grid_dist)1.0*(grid_dist>=(0.5*opt.m)); % For learning, do not allow overlapping regions
%opt.spatial_push = [];
%disp(opt);
opt = struct_merge(opt, input_opts); % add the inputted changes
[opt, ROI_mask, ROIs] = chomp(opt);


 %
%Inferring with 1200 cells given the supervised learned subspace
opt.cells_per_image = 1200;
opt.init_iter = opt.niter; %Just running the inference
opt.niter = opt.niter+1; %Just running the inference
%opt.spatial_push = @(grid_dist)1.0*(grid_dist>=(opt.m/3));
%opt.spatial_push = @(grid_dist)logsig(0.7*grid_dist-floor(opt.m-1)).*(grid_dist>=(2*opt.m/3)); %@(grid_dist, sharp)logsig(sharp*grid_dist-floor(sharp*2*obj.m/2-1));
% % opt.spatial_push = [];
% %opt.spatial_push = @(grid_dist)1.0*(grid_dist>=(opt.m/3)); % For inference, be less restrictive on spatial push
% %opt.spatial_push = @(grid_dist)logsig(0.5*grid_dist-floor(opt.m/2-1)); %@(grid_dist, sharp)logsig(sharp*grid_dist-floor(sharp*2*obj.m/2-1));
% %opt.spatial_push = @(grid_dist)(2*(logsig(grid_dist/(opt.m/8))-0.5).*(grid_dist<(1.5*opt.m))+(grid_dist>=(1.5*opt.m)));
% opt.spatial_push = @(grid_dist)1.0*(grid_dist>=(opt.m/2));
[opt, ROI_mask, ROIs] = chomp(opt);
saved_opts{end+1} = struct_merge(chomp_options(),opt);
get_neurofinder_results( opt, ROIs, true );


%get_cell_timeseries(opt);
% 
% Learn the classifier and ROI regressors given the output of CHOMP and training data
%trainedModel = train_neurofinder_ROIs( opt, true );


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
cur_init_model = {};
for objs = 1:opt.NSS, cur_init_model{objs} = 'given'; end;
opt.init_model = cur_init_model;
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
saved_opts{end+1} = struct_merge(chomp_options(),opt);
%[ROI_mask, ROIs] = get_neurofinder_ROIs( opt, trainedModel );

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

% Mask out weird areas in the corners (cancel for now)
% opt.mask = 1;
% opt.mask_image = ones([512,512]);
% opt.mask_image(:,1:round(cell_max_size/2.)) = 0;
% opt.mask_image(:,(end-round(cell_max_size/2.)):end) = 0;
% opt.mask_image(1:round(cell_max_size/2.),:) = 0;
% opt.mask_image((end-round(cell_max_size/2.)):end, :) = 0;
% opt.mask_image(1:70,1:70) = 0; % top left
% opt.mask_image(350:end,1:150) = 0; % bottom left
% opt.mask_image(1:140,300:end) = 0; % top right
% opt.mask_image= imresize(opt.mask_image, opt.spatial_scale,'nearest');


[~, tmp1, tmp2] = fileparts(fileparts(fileparts(fileparts(opt.data_path)))); % Important if you want to have nice file prefixes (corresponding to main folder name)
opt.file_prefix = [tmp1 tmp2 '_preproc2P' preproc2P_stamp];


[opt, ROI_mask, ROIs] = chomp(opt);
saved_opts{end+1} = struct_merge(chomp_options(),opt);
%[ROI_mask, ROIs] = get_neurofinder_ROIs( opt, trainedModel );

ROI_to_json(opt, ROIs);

out_timestamp = opt.timestamp;

end