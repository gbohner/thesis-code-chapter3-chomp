% Run all of neurofinder files with given timestamps and given (custom)
% options

cur_timestamp = datestr(now(),30);

input_opts = struct();
input_opts.timestamp = cur_timestamp;

% Preproc options
input_opts.spatial_scale = 1.0;
input_opts.time_scale = 1.0;
input_opts.subtract_background = 1;
input_opts.denoise_medianfilter = 0;
input_opts.whiten = 0;

% Inference/learning options
input_opts.mom = 1;
input_opts.mom_weights = [];
input_opts.NSS = 1;
input_opts.KS = 3;
input_opts.init_model = 'neurofinder_preproc2P';%; %'donut_four';%{'donut_conv_new', 'filled'};%'neurofinder_preproc2P';%{'donut_conv'}; %{'donut_conv', 'pointlike'};%{'donut_four', 'pointlike'}; %'donut_two'; %'donut_conv'; %'pointlike'; %'filled'; %{'filled', 'donut'};
input_opts.niter = 1; 
input_opts.learn = 0; %input_opts.niter =5;
input_opts.learn_decomp = 'COV';%'COV_RAW'; % COV, NMF
input_opts.diag_tensors = 1; %1;
input_opts.diag_cumulants = 1;
input_opts.diag_cumulants_offdiagonly = 0;
input_opts.W_addflat = 0;
input_opts.W_force_round = 1;
input_opts.local_peaks = 0;
input_opts.blank_reconstructed = 1;
input_opts.W_weight_type = 'decomp'; %'decomp'; % 'uniform' / 'decomp' - latter using SVD weights to penalise use of bases (but only works with diag_tensors=1 for now)

% ROI options
input_opts.ROI_type = 'ones_origsize';%'quantile_dynamic_origsize';
input_opts.ROI_params = 0.5;




%% Run all datasets with the same settings

preproc_stamp = '_expertPrior_unampLik_gitsha_2bd0d72_evalgit_db4ade8_rPC_1_origPMgain_useNans_targetCoverage_10_grid_30_7';

savename = ['./Examples/saved_runs_neurofinder_preproc_1_nowhiten_' cur_timestamp '.mat'];


all_saved_opts = {};
all_training_results = {};

[~,cur_saved_opts] = func_dataset_00_preproc2P(preproc_stamp, input_opts);
all_saved_opts{end+1} = cur_saved_opts;
[ROI_image, ROIs] = getROIs( cur_saved_opts{1} );
all_training_results{end+1} = get_neurofinder_results( cur_saved_opts{1}, ROIs, 1 );
save(savename, 'all_saved_opts', 'all_training_results');

[~,cur_saved_opts] = func_dataset_01_preproc2P(preproc_stamp, input_opts);
all_saved_opts{end+1} = cur_saved_opts;
[ROI_image, ROIs] = getROIs( cur_saved_opts{1} );
all_training_results{end+1} = get_neurofinder_results( cur_saved_opts{1}, ROIs, 1 );
save(savename, 'all_saved_opts', 'all_training_results');

[~,cur_saved_opts] = func_dataset_02_preproc2P(preproc_stamp, input_opts);
all_saved_opts{end+1} = cur_saved_opts;
[ROI_image, ROIs] = getROIs( cur_saved_opts{1} );
all_training_results{end+1} = get_neurofinder_results( cur_saved_opts{1}, ROIs, 1 );
save(savename, 'all_saved_opts', 'all_training_results');

[~,cur_saved_opts] = func_dataset_03_preproc2P(preproc_stamp, input_opts);
all_saved_opts{end+1} = cur_saved_opts;
[ROI_image, ROIs] = getROIs( cur_saved_opts{1} );
all_training_results{end+1} = get_neurofinder_results( cur_saved_opts{1}, ROIs, 1 );
save(savename, 'all_saved_opts', 'all_training_results');

[~,cur_saved_opts] = func_dataset_04_00_preproc2P(preproc_stamp, input_opts);
all_saved_opts{end+1} = cur_saved_opts;
[ROI_image, ROIs] = getROIs( cur_saved_opts{1} );
all_training_results{end+1} = get_neurofinder_results( cur_saved_opts{1}, ROIs, 1 );
save(savename, 'all_saved_opts', 'all_training_results');

% [~,cur_saved_opts] = func_dataset_00_preproc2P(preproc_stamp, input_opts);
% all_saved_opts(end+1) = cur_saved_opts;
