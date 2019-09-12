do_figure_exports = 1;
if do_figure_exports
  clear all;
  do_figure_exports = 1;
  addpath('export_fig-master/');
  fig_timestamp = ['_' datestr(now,30)];
  %fig_timestamp = '_20190605T171329';
  %fig_timestamp = '_20190530T132106';
  %fig_path = '/Users/gergobohner/Dropbox (Personal)/Gatsby/Thesis/Code/Ch2-Matlab-figures/Saved_figures/';
  fig_path = '/Users/gergobohner/Dropbox (Personal)/Gatsby/Thesis/Text_copy/Figures/Chapter2/Final_figures/';
  f_set_axis_props = @(axis_handle)set(axis_handle,'FontSize',16);
  f_getfigfullpath= @(fname)[fig_path fname fig_timestamp];
  f_export_fig = @(fname)export_fig(f_getfigfullpath(fname), '-pdf', '-nocrop', '-q101');
  f_export_fig_raster=@(fname)print(f_export_fig_raster_crop(gcf),'-dpdf',f_getfigfullpath(fname)); % Check if cropping is needed, but dont think so
  %f_export_fig = @(fname)eval("export_fig eval([fig_path fname fig_timestamp]) '-pdf', '-nocrop', '-q101'");
end

CHOMP_code_dir = '/mnt/gatsby/nfs/nhome/live/gbohner/Dropbox_u435d_unsynced_20190502/Gatsby/Research/CHOMP/';
addpath(genpath(CHOMP_code_dir));


my_colormaps = load([CHOMP_code_dir 'Examples/matlab_colormaps.mat']);
cur_colormap = my_colormaps.felfire;
set(0, 'DefaultFigureColormap',  cur_colormap)

my_new_figure_image_wrapper = @(fig_handle)set(fig_handle,...
  'Color', 'none', 'Units','pixels','Position',[100,100,800,600],...
  'Colormap',cur_colormap);
my_new_figure_image = @(fig_num)my_new_figure_image_wrapper(figure(fig_num));

% model = load(['/mnt/gatsby' get_path(all_saved_opts{1}{1},'output_iter',all_saved_opts{1}{1}.niter)])
%update_visualize_model(model, 1:400);


% Load original space model (for mean / variance image)
orig_model_01 = load('/mnt/gatsby/nfs/data/gergo/Neurofinder_update/neurofinder.01.00/CHOMP/output/neurofinder.01.00_20170727T144747_iter_3.mat');

% preProcessing params
prior = 'expertPrior';%prior = 'noPrior';
lik = 'unampLik'; %lik = 'linLik';
gitsha = 'gitsha_2bd0d72_evalgit_db4ade8';
trainType = '_rPC_1_origPMgain_useNans';
targetCoverage = '_targetCoverage_10';
gridType = '_grid_30_7';

% Load preprocessed model fit
dataset_name = 'neurofinder.02.00';
% Different iters are with different settings!
chomp_iter = '2'; % diag_cumulants =0, diag_cumulants_extradiagonal =1
chomp_timestamp = '20190529T060454'; % after recent fixes % .  iter 2 extra-diagonal, iter 3-diag_cumulant, iter 4 - neither

stamp = ['_preproc2P_' prior '_' lik '_' gitsha trainType targetCoverage gridType];

cur_model = load(['/mnt/gatsby/nfs/data/gergo/Neurofinder_update/' ...
    dataset_name '/preproc2P/CHOMP/output/' dataset_name ...
    stamp '_' ...
    chomp_timestamp '_iter_' chomp_iter '.mat']);

model = cur_model.model;
model.opt.root_folder = '/mnt/gatsby';
opt = model.opt;


%update_visualize_model(model, use_cells);
  

%% Do figures

% Plot mean and variance pixelwise whole image
my_new_figure_image(4); clf; imagesc(model.y_orig); axis image; axis off; colorbar; 
if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_mean_green'); end

my_new_figure_image(5); clf; imagesc(model.opt.cumulant_pixelwise{2}); axis image; axis off; colorbar; colormap(my_colormaps.fire)
if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_var_red'); end

load([opt.root_folder, fileparts(opt.data_path) '/spatial_gain.mat']); % loads 'spatial_gain' image
stretch_factor = csvread([opt.root_folder, fileparts(opt.data_path) '/stretch_factor.csv']); % get stretch factor

my_new_figure_image(6); clf; imagesc(spatial_gain); axis image; axis off; colorbar; colormap(my_colormaps.div)
if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_spatial_gain'); end

my_new_figure_image(7); clf; imagesc((model.y_orig.*spatial_gain)./stretch_factor); axis image; axis off; colorbar; colormap(my_colormaps.felfire)
if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_mean_uncorr'); end

my_new_figure_image(8); clf; imagesc((model.opt.cumulant_pixelwise{2}.*spatial_gain.^2./stretch_factor.^2)); axis image; axis off; colorbar; colormap(my_colormaps.fire)
if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_var_uncorr'); end


%% Zoomed in areas
marker_size_orig = 30;
marker_size_model = 50;
marker_linewidth_model = 2.5;
marker_alpha_model = 1.0;


% helper funcs to plot zoomed in area
get_cur_H_inds = @(H,cur_area)(H(:,1)>cur_area(1) & H(:,1)<cur_area(3) & H(:,2)>cur_area(2) & H(:,2)<cur_area(4));
get_cur_H = @(H, cur_area)([H(get_cur_H_inds(H,cur_area),1)-cur_area(1),H(get_cur_H_inds(H,cur_area),2)-cur_area(2)]);
get_area_inds = @(cur_area){cur_area(1):cur_area(3), cur_area(2):cur_area(4)};
get_area_rect = @(cur_area)[cur_area(2), cur_area(1), cur_area(4)-cur_area(2), cur_area(3)-cur_area(1)];


if chomp_iter == '2'
  use_cells = [60,68,140,108,109,110,111,140]; % for clustered cells chomp_timestamp = '20190529T060454';  chomp_iter = '2'; % after recent fixes
  reconst_type = '_diag_offdiag';
elseif chomp_iter =='3'
  %TODO
  reconst_type = '_diag_only';
elseif chomp_iter =='4' 
  use_cells = [67,79,125:128]; % for single cells chomp_timestamp = '20190529T060454';  chomp_iter = '4'; % after recent fixes
  reconst_type = '_offdiag_only';
end


H = model.H(use_cells,:);

%cur_area = [min(H(:,1:2),[],1)-((opt.m-1)/2), max(H(:,1:2),[],1)+((opt.m-1)/2)];
cur_area = [72,   188,   120,   249];

cur_zoomed_areas = {cur_area};
cur_border_colors = {'w'};

for i1 = 1:length(cur_zoomed_areas)
  cur_area = cur_zoomed_areas{i1}; cur_border_color = cur_border_colors{i1};
  cur_area_inds = get_area_inds(cur_area);
  cur_model_H = get_cur_H(H,cur_area);
  %cur_orig_H = get_cur_H(orig_H,cur_area);
  my_new_figure_image(50+i1); clf; imagesc(model.y_orig(cur_area_inds{1},cur_area_inds{2})); axis image; axis off; colorbar
%   hold on; scatter(cur_model_H(:,2), cur_model_H(:,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
%   hold on; scatter(cur_orig_H(:,2), cur_orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
%   hold on; rectangle('Position',[0,0,length(cur_area_inds{2})+1,length(cur_area_inds{1})+1], 'EdgeColor',cur_border_color,'Linewidth',10.);
%   hold on; rectangle('Position',[-2,-2,length(cur_area_inds{2})+5,length(cur_area_inds{1})+5], 'EdgeColor','k','Linewidth',1.);
  
  if do_figure_exports, f_set_axis_props(gca); f_export_fig(['fig_ch2_final_example_01_data_02_results_image_zoom_' num2str(i1)]); end
  
  % Show what if we added more cells
  %cur_model_H = get_cur_H(model.H(use_cells_double,:),cur_area);
  %hold on; scatter(cur_model_H(:,2), cur_model_H(:,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
  %if do_figure_exports, f_set_axis_props(gca); f_export_fig(['fig_ch2_data_02_results_image_zoom_' num2str(i1) '_morecells']); end
  
  
  % put border on the original images
  figure(4); hold on; rectangle('Position',get_area_rect(cur_area), 'EdgeColor',cur_border_color,'Linewidth',1.);
  figure(5); hold on; rectangle('Position',get_area_rect(cur_area), 'EdgeColor',cur_border_color,'Linewidth',1.);
end

figure(4);
if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_mean_green_zoombox'); end
figure(5);
if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_var_red_zoombox'); end


%% Do the variance plots
% Run get_reconst_example_2 on the cluster to get this array
%load(['~/Data/CHOMP_Examples/reconst_example_' chomp_timestamp '_iter_' chomp_iter '.mat'])
load('~/Data/CHOMP_Examples/reconst_example_neurofinder.02.00_20190529T010717.mat'); % this is on iter 2
tmp = load('~/Data/CHOMP_Examples/reconst_example_neurofinder.02.00_20190529T010717.mat');



%%

% Show the variance image
inds = get_area_inds(cur_area);

mom1 = 1;
my_new_figure_image(100+mom1*10+1); clf; imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})); axis image; colorbar; colormap(my_colormaps.felfire)
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_mean_zoom'); end


cur_colormap = my_colormaps.fire;

mom1 = 2;
my_new_figure_image(100+mom1*10+1); clf; imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})); axis image; colorbar; colormap(cur_colormap)
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_variance_zoom'); end

%% Line covariance (vertical)
% Get a single line of the variance and thus covariance
mom1 =2;
patch_mask = zeros(szPatch(1),szPatch(2));
patch_mask(:,17) = 1;

% Add the patch-mask as a line of black onto the image;
my_new_figure_image(100+mom1*10+1); clf; imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})); axis image; colorbar; colormap(cur_colormap)
hold on; line([17,17],[1,49], 'LineWidth', 2., 'Color', 'black' )
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_variance_vert_zoom_line'); end



full_var = opt.cumulant_pixelwise{2}(inds{1},inds{2});
line_var = full_var(logical(patch_mask(:)));
my_new_figure_image(101); clf;
plot(line_var, ...
  'LineWidth', 3., 'Color', [0 0 0]...
  )
ylabel('Variance (a.u.)')
xlabel('Y-Location (pixel)')
f_set_axis_props(gca);
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_line_vert_variance_plot'); end

patch_mask_outer = patch_mask(:)*patch_mask(:)';
line_covar = data_cov(logical(patch_mask_outer(:)));
%figure; imagesc(reshape(line_covar,sum(patch_mask(:)),sum(patch_mask(:))));


line_covar = reshape(line_covar,sum(patch_mask(:)),sum(patch_mask(:)));
clim_covar = [min(line_covar(:)), 4e6];
my_new_figure_image(102); clf; imagesc(line_covar); colormap(cur_colormap); axis image; colorbar
if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_final_example_01_data_02_preproc_line_vert_covariance_image_raw_noclim'); end
my_new_figure_image(102); clf; imagesc(line_covar, clim_covar); colormap(cur_colormap); axis image; colorbar
if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_final_example_01_data_02_preproc_line_vert_covariance_image_raw'); end
my_new_figure_image(1021); clf; imagesc(gauss_smooth_matrix(line_covar,1.2,1.,1.),clim_covar); colormap(cur_colormap); colorbar; axis image

cell_locs = {[6,22], [10,30]};
cur_border_colors = {'m','b'};
for cell1 = 1:length(cell_locs)
  figure(1021); hold on;
  rectangle('Position',get_area_rect([cell_locs{cell1}(1),cell_locs{cell1}(1),cell_locs{cell1}(2), cell_locs{cell1}(2)]),...
    'EdgeColor',cur_border_colors{cell1},'Linewidth',2.);
  figure(100+mom1*10+1); hold on; 
  line([17-cell1,17-cell1],cell_locs{cell1}, 'LineWidth', 2., 'Color', cur_border_colors{cell1} )
end

figure(1021);
if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_final_example_01_data_02_preproc_line_vert_covariance_image_processed'); end

figure(100+mom1*10+1);
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_variance_vert_zoom_line_colored'); end


% Significant correlation in horizontal direction but not in 
spatial_falloff = [];
for offdiag = 1:10
 spatial_falloff(offdiag) = 1./nanmedian(diag(line_covar)./[(diag(line_covar,offdiag)+diag(line_covar,-offdiag))./2.;nan*ones(offdiag,1)]);
end
my_new_figure_image(1792); clf; hold on; plot(0:length(spatial_falloff), [1.,spatial_falloff], ...
  'LineWidth', 3., 'Color', [0 0 0]...
  )

%% Line covariance (horizontal)
% Get a single line of the variance and thus covariance
mom = 2;
patch_mask = zeros(szPatch(1),szPatch(2));
patch_mask(19,:) = 1;

% Add the patch-mask as a line of black onto the image;
my_new_figure_image(100+mom1*10+1); clf; imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})); axis image; colorbar; colormap(cur_colormap)
hold on; line([1,62],[19,19], 'LineWidth', 2., 'Color', 'black' )
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_variance_horiz_zoom_line'); end



full_var = opt.cumulant_pixelwise{2}(inds{1},inds{2});
line_var = full_var(logical(patch_mask(:)));
my_new_figure_image(101); clf;
plot(line_var, ...
  'LineWidth', 3., 'Color', [0 0 0]...
  )
ylabel('Variance (a.u.)')
xlabel('Y-Location (pixel)')
f_set_axis_props(gca);
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_line_horiz_variance_plot'); end

patch_mask_outer = patch_mask(:)*patch_mask(:)';
line_covar = data_cov(logical(patch_mask_outer(:)));
%figure; imagesc(reshape(line_covar,sum(patch_mask(:)),sum(patch_mask(:))));


line_covar = reshape(line_covar,sum(patch_mask(:)),sum(patch_mask(:)));
%clim_covar = [min(line_covar(:)), 4e6];
clim_covar = [min(line_covar(:)), 7e6];
my_new_figure_image(102); clf; imagesc(line_covar); colormap(cur_colormap); axis image; colorbar
if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_final_example_01_data_02_preproc_line_horiz_covariance_image_raw_noclim'); end
my_new_figure_image(102); imagesc(line_covar, clim_covar); colormap(cur_colormap); axis image; colorbar;
if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_final_example_01_data_02_preproc_line_horiz_covariance_image_raw'); end
%my_new_figure_image(1021); clf; imagesc(gauss_smooth_matrix(line_covar,1.2,1.,1.),clim_covar); colormap(cur_colormap); colorbar; axis image

my_new_figure_image(1021); clf; imagesc(gauss_smooth_matrix(line_covar,1.2,4.,1.), clim_covar); colormap(cur_colormap); colorbar; axis image

cell_locs = {[33,53], [37,58]};
for cell1 = 1:length(cell_locs)
  figure(1021); hold on;
  rectangle('Position',get_area_rect([cell_locs{cell1}(1),cell_locs{cell1}(1),cell_locs{cell1}(2), cell_locs{cell1}(2)]),...
    'EdgeColor',cur_border_colors{cell1},'Linewidth',2.);
  figure(100+mom1*10+1); hold on; 
  line(cell_locs{cell1},[19-cell1,19-cell1], 'LineWidth', 2., 'Color', cur_border_colors{cell1} )
end
%my_new_figure_image(102); imagesc(line_covar, clim_covar); colormap(cur_colormap);
%my_new_figure_image(1021); imagesc(gauss_smooth_matrix(line_covar,1.2,1.)); colormap(cur_colormap); colorbar
figure(1021);
if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_final_example_01_data_02_preproc_line_horiz_covariance_image_processed'); end

figure(100+mom1*10+1);
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_variance_horiz_zoom_line_colored'); end


% Significant correlation in horizontal direction but not in 
spatial_falloff = [];
for offdiag = 1:10
 spatial_falloff(offdiag) = 1./nanmedian(diag(line_covar)./[(diag(line_covar,offdiag)+diag(line_covar,-offdiag))./2.;nan*ones(offdiag,1)]);
end
figure(1792); hold on; plot(0:length(spatial_falloff), [1.,spatial_falloff], ...
  'LineWidth', 3., 'Color', [1.0000    0.4980    0.0549]... % dark orange
  )
xlabel('Distance (pixel)')
ylabel('Median correlation')
legend({'Vertical (non-scanning)', 'Horizontal (scanning)'})
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_variance_XYfalloff'); end



%% Box covariance


my_new_figure_image(100+mom1*10+1); clf; imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})); axis image; colorbar; colormap(cur_colormap)
%if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_variance_zoom_box'); end


% Get a box of the variance and thus covariance
patch_mask = zeros(szPatch(1),szPatch(2));
patch_mask(5:25,45:60) = 1;

% Add the patch-mask as a line of black onto the image;
hold on; rectangle('Position', [45,5, 15,20], 'LineWidth', 2., 'EdgeColor', 'black' )
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_variance_zoom_box'); end



patch_mask_outer = patch_mask(:)*patch_mask(:)';
box_covar = data_cov(logical(patch_mask_outer(:)));
box_covar = reshape(box_covar,sum(patch_mask(:)),sum(patch_mask(:)));

my_new_figure_image(103); clf; imagesc(box_covar); colormap(cur_colormap); colorbar; axis image 
if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_final_example_01_data_02_preproc_box_covariance_image_raw_noclim'); end
my_new_figure_image(103); clf; imagesc(box_covar, clim_covar); colormap(cur_colormap); colorbar; axis image 
if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_final_example_01_data_02_preproc_box_covariance_image_raw'); end

%my_new_figure_image(103); imagesc(gauss_smooth_matrix(box_covar,2.,1.,1.)); colormap(cur_colormap);
my_new_figure_image(103); clf; imagesc(gauss_smooth_matrix(box_covar,2.0,[22,43,64],1.),clim_covar); colormap(cur_colormap); colorbar; axis image

% Draw vertical lines

% Line1 
figure(100+mom1*10+1); hold on; 
line([47,47],[10,24], 'LineWidth', 2., 'Color', cur_border_colors{1} )
figure(103); hold on;
rectangle('Position',get_area_rect([(47-45)*21+(10-5),(47-45)*21+(10-5),(47-45)*21+(24-5),(47-45)*21+(24-5)]),...
  'EdgeColor',cur_border_colors{1},'Linewidth',2.);

% Line2 
figure(100+mom1*10+1); hold on; 
line([55,55],[10,24], 'LineWidth', 2., 'Color', cur_border_colors{2} )
figure(103); hold on;
rectangle('Position',get_area_rect([(55-45)*21+(10-5),(55-45)*21+(10-5),(55-45)*21+(24-5),(55-45)*21+(24-5)]),...
  'EdgeColor',cur_border_colors{2},'Linewidth',2.);

figure(103);
if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_final_example_01_data_02_preproc_box_covariance_image_processed'); end

figure(100+mom1*10+1);
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_variance_zoom_box_colored'); end


% rectangle('Position',get_area_rect([cell_locs{cell1}(1),cell_locs{cell1}(1),cell_locs{cell1}(2), cell_locs{cell1}(2)]),...
%   'EdgeColor',cur_border_colors{cell1},'Linewidth',2.);



% %% Full covariance
% 
% clim_covar = [0, 4e6];
% my_new_figure_image(200); imagesc(data_cov, clim_covar); colormap(my_colormaps.fire);
% if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_data_02_example_line_covariance_image'); end


%% Skewness and kurtosis

% Load co-skewness / co-kurtosis example (xcums)
load('/Users/gergobohner/Data/CHOMP_Examples/data01_xcum_examples_20190605_large.mat') % loads xcums
%
% squeeze(inp.data.proc_stack.Y(72:99,232,:)); OR zoomed (1:27,45) vertical
% line

cur_colormap = my_colormaps.fire;


xcum_subind = [6:10];

my_new_figure_image(711); imagesc(xcums{1}(xcum_subind)); axis image; colorbar
if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_final_example_01_data_02_preproc_higher_ex_mom1'); end
my_new_figure_image(712); imagesc(xcums{2}(xcum_subind,xcum_subind)); axis image; colorbar; colormap(cur_colormap)
if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_final_example_01_data_02_preproc_higher_ex_mom2'); end
my_new_figure_image(713); imagesc(reshape(xcums{3}(xcum_subind,xcum_subind,xcum_subind),numel(xcum_subind).^2,[])); axis image; colorbar; colormap(cur_colormap)
if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_final_example_01_data_02_preproc_higher_ex_mom3'); end
my_new_figure_image(714); imagesc(reshape(xcums{4}(xcum_subind,xcum_subind,xcum_subind,xcum_subind),numel(xcum_subind).^2,[])); axis image; colorbar; colormap(cur_colormap)
if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_final_example_01_data_02_preproc_higher_ex_mom4'); end

% xcums4_cur = xcums{4}./mply(xcums{2},shiftdim(xcums{2},-1),1);
% xcums4_cur = xcums{4};
% figure; imagesc(reshape(xcums4_cur(xcum_subind,xcum_subind, xcum_subind,xcum_subind),numel(xcum_subind).^2,[])...
%   ,[-100,100]); axis image; colormap(cur_colormap); colorbar
% 
% figure; imagesc(gauss_smooth_matrix(xcums{2}(xcum_subind,xcum_subind),1.0,1,1.))
% figure; imagesc(gauss_smooth_matrix(reshape(xcums4_cur(xcum_subind,xcum_subind, xcum_subind,xcum_subind),numel(xcum_subind).^2,[]),...
%   1.5,1,1.)); colormap(cur_colormap); colorbar
% figure; imagesc(gauss_smooth_matrix(reshape(xcums4_cur(xcum_subind,xcum_subind, xcum_subind,xcum_subind),numel(xcum_subind).^2,[]),...
%   4.5,1,1.),[-10,10]); colormap(cur_colormap); colorbar

%% Small example

load('/Users/gergobohner/Data/CHOMP_Examples/data01_xcum_examples_20190605.mat') % loads xcums (4 pixels only)

cur_colormap = my_colormaps.fire;
x = xcums{1}(:);
figure; imagesc(xcums{1}(:)); colorbar; axis image; colormap(my_colormaps.felfire)
figure; imagesc(reshape(xcums{2},[],size(x,1))); colorbar; axis image; colormap(cur_colormap)
figure; imagesc(reshape(xcums{3},[],size(x,1))); colorbar; axis image; colormap(cur_colormap)
figure; imagesc(reshape(xcums{4},[],size(x,1)*size(x,1))); colorbar; axis image; colormap(cur_colormap)
%figure; imagesc(reshape(xcums{4}./symmetrise(mply(xcums{2},shiftdim(xcums{2},-1),1)),[],size(x,1)*size(x,1)),[-10,10]); colorbar



%% Skewness and kurtosis

%Pixelwise correction

mom1 = 3;
% my_new_figure_image(300+mom1*10+1); clf; imagesc(opt.cumulant_pixelwise{mom1}.*spatial_gain.^mom1./((opt.cumulant_pixelwise{2}.*spatial_gain.^(2)).^(mom1./2.))); axis image; colorbar; colormap(cur_colormap)
% if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_skewness_uncorr'); end
my_new_figure_image(300+mom1*10+1); clf; imagesc(opt.cumulant_pixelwise{mom1}./(opt.cumulant_pixelwise{2}.^(mom1./2.)), [-6,6]); axis image; colorbar; colormap(cur_colormap)
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_skewness'); end

mom1 = 4;
% my_new_figure_image(300+mom1*10+22); clf; imagesc(opt.cumulant_pixelwise{mom1}.*spatial_gain.^(mom1)./((opt.cumulant_pixelwise{2}.*spatial_gain.^(2)).^(mom1./2.)), [-6,6]); axis image; colorbar; colormap(cur_colormap)
% if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_kurtosis_uncorr'); end
my_new_figure_image(300+mom1*10+1); clf; imagesc(opt.cumulant_pixelwise{mom1}./(opt.cumulant_pixelwise{2}.^(mom1./2.)), [-10,10]); axis image; colorbar; colormap(cur_colormap)
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_final_example_01_data_02_preproc_kurtosis'); end

%%

cur_colormap = my_colormaps.div;

mom1 = 3;
my_new_figure_image(100+mom1*10+1); clf; imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2}).*spatial_gain(inds{1},inds{2}).^mom1); axis image; colorbar; colormap(cur_colormap)
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_data_02_example_variance_zoom_line_horiz'); end

mom1 = 4;
my_new_figure_image(100+mom1*10+1); clf; imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2}).*spatial_gain(inds{1},inds{2}).^mom1); axis image; colorbar; colormap(cur_colormap)
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_data_02_example_variance_zoom_line_horiz'); end


