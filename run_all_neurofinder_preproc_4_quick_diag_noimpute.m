% Run all of neurofinder files with given timestamps and given (custom)
% options

cur_timestamp = datestr(now(),30);

input_opts = struct();
input_opts.timestamp = cur_timestamp;

% Preproc options
input_opts.spatial_scale = 0.3;
input_opts.time_scale = 0.5; % avoid time correlated noise/signal (we only want spatial correlation)
input_opts.subtract_background = 1;
input_opts.denoise_medianfilter = 0;
input_opts.whiten = 0;

% Inference/learning options
input_opts.mom = 4;
input_opts.mom_weights = [0.,0.,0.,1.];
input_opts.NSS = 1;
input_opts.KS = 8;
input_opts.init_model = 'neurofinder_preproc2P';%'donut_four';%; %'donut_four';%{'donut_conv_new', 'filled'};%'neurofinder_preproc2P';%{'donut_conv'}; %{'donut_conv', 'pointlike'};%{'donut_four', 'pointlike'}; %'donut_two'; %'donut_conv'; %'pointlike'; %'filled'; %{'filled', 'donut'};
input_opts.niter = 1; 
input_opts.learn = 0; %input_opts.niter =5;
input_opts.cells_per_image = 5;
input_opts.learn_decomp = 'COV_RAW';%'COV_RAW'; % COV, NMF
input_opts.diag_tensors = 1; % for now 1 works better it seems?
input_opts.diag_cumulants = 1;
input_opts.standardise_cumulants = 1; % test if it works with multivariate!
input_opts.diag_cumulants_offdiagonly = 0;
input_opts.diag_cumulants_extradiagonal = 0;
input_opts.W_addflat = 0;
input_opts.W_force_round = 1;
input_opts.local_peaks = 0;
input_opts.local_peaks_size = [15,7,0];
input_opts.local_peaks_gauss = 3.;
input_opts.blank_reconstructed = 0;
input_opts.spatial_push = @(x)1.*(x>=15);
input_opts.W_weight_type = 'uniform'; %'decomp'; % 'uniform' / 'decomp' - latter using SVD weights to penalise use of bases (but only works with diag_tensors=1 for now)
input_opts.reconst_upto_median_WY = 1;
input_opts.spatial_gain = 1; % If scalar, find and load a spatial gain matrix (the result of preproc) (otherwise supply the matrix)
input_opts.spatial_gain_undo_raw = 1; % If 1, data.proc_stack will be multiplied by the supplied spatial gain
input_opts.stretch_factor = 1; % If 1 - load the stretch factor from preproc. If >0, but ~=1, keep it as scalar
input_opts.spatial_gain_renormalise_lik = 1; % If spatial gain is supplied, use it to renormalise likelihood
input_opts.reconst_upto_median_WY = 1; % If 1, for each dimension of WY, gets the median
input_opts.zeros_ignore = 1;
input_opts.subtract_background = 0; % when using background level and reconst_to_median this is better not done

% ROI options
input_opts.ROI_type = 'ones_origsize';%'quantile_dynamic_origsize';
input_opts.ROI_params = 0.5;




%% Run all datasets with the same settings

preproc_stamp = '_expertPrior_unampLik_gitsha_2bd0d72_evalgit_db4ade8_rPC_1_origPMgain_useNans_targetCoverage_10_grid_30_7';
preproc_stamp_noimpute = '_expertPrior_unampLik_gitsha_2bd0d72_evalgit_db4ade8_noimpute_rPC_1_origPMgain_useNans_targetCoverage_10_grid_30_7';
preproc_stamp = preproc_stamp_noimpute; % use unimputed data

savename = ['./Examples/saved_runs_' mfilename(9:end-2)  '_' cur_timestamp '.mat'];


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
