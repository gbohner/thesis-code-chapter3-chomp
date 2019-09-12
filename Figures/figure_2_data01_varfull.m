do_figure_exports = 0;
if do_figure_exports
  clear all;
  do_figure_exports = 1;
  addpath('export_fig-master/');
  %fig_timestamp = ['_' datestr(now,30)];
  fig_timestamp = '_20190608T172246';
  %fig_timestamp = '_20190605T171329';
  %fig_timestamp = '_20190530T132106';
  %fig_path = '/Users/gergobohner/Dropbox (Personal)/Gatsby/Thesis/Code/Ch2-Matlab-figures/Saved_figures/';
  fig_path = '/Users/gergobohner/Dropbox (Personal)/Gatsby/Thesis/Text_copy/Figures/Chapter2/Neurofinder_results_CHOMP/';
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
dataset_name = 'neurofinder.01.00';
% Different iters are with different settings!
chomp_iter = '2'; % diag_cumulants =0, diag_cumulants_extradiagonal =1
chomp_iter = '3'; % diag_cumulants =1, diag_cumulants_extradiagonal =0
chomp_iter = '4'; % diag_cumulants =0, diag_cumulants_extradiagonal =0
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
  
  
%% Get ground turth ROIs
  
[orig_ROIs, orig_H, orig_ROI_mask] = get_neurofinder_orig_ROIs(opt);


%% Do figures

% Show original model and images
use_cells = 1:250;%size(orig_H,1);
use_cells_double = use_cells;%1:(2*size(orig_H,1));


% Plot settings for markers
marker_size_orig = 15;
marker_size_model = 30;
marker_color_orig = 'b';
marker_color_model = [1.0000    0.4980    0.0549];
marker_alpha_orig = 1.0;
marker_alpha_model= 0.8;
marker_linewidth_model = 1.5; 


my_new_figure_image(1); clf; imagesc(imresize(orig_model_01.model.V,1./orig_model_01.model.opt.spatial_scale)); axis image; axis off; colorbar;
hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_data_01_orig_image_gray'); end

my_new_figure_image(2); clf; imagesc(model.opt.cumulant_pixelwise{2}); axis image; axis off; colorbar; 
hold on; scatter(model.H(use_cells,2), model.H(use_cells,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);



my_new_figure_image(3); clf; imagesc(orig_model_01.model.y_orig); axis image; axis off; colorbar; colormap gray
hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_data_01_orig_image_gray'); end


my_new_figure_image(4); clf; imagesc(orig_model_01.model.y_orig); axis image; axis off; colorbar; 
%hold on; scatter(model.H(use_cells,2), model.H(use_cells,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
%hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'MarkerEdgeColor','w','Linewidth',0.5,'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_data_01_orig_image_green'); end



my_new_figure_image(5); clf; imagesc(model.y_orig); axis image; axis off; colorbar; 
hold on; scatter(model.H(use_cells,2), model.H(use_cells,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);

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
cur_area = [72,   188,   120,   249]; % iter 4

cur_zoomed_areas = {cur_area};
cur_border_colors = {'w'};

for i1 = 1:length(cur_zoomed_areas)
  cur_area = cur_zoomed_areas{i1}; cur_border_color = cur_border_colors{i1};
  cur_area_inds = get_area_inds(cur_area);
  cur_model_H = get_cur_H(H,cur_area);
  cur_orig_H = get_cur_H(orig_H,cur_area);
  my_new_figure_image(50+i1); clf; imagesc(model.y_orig(cur_area_inds{1},cur_area_inds{2})); axis image; axis off; colorbar
  hold on; scatter(cur_model_H(:,2), cur_model_H(:,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
  hold on; scatter(cur_orig_H(:,2), cur_orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
  hold on; rectangle('Position',[0,0,length(cur_area_inds{2})+1,length(cur_area_inds{1})+1], 'EdgeColor',cur_border_color,'Linewidth',10.);
  hold on; rectangle('Position',[-2,-2,length(cur_area_inds{2})+5,length(cur_area_inds{1})+5], 'EdgeColor','k','Linewidth',1.);
  
  if do_figure_exports, f_set_axis_props(gca); f_export_fig(['fig_ch2_data_01_results_image_zoom_' num2str(i1)]); end
  
  % Show what if we added more cells
  cur_model_H = get_cur_H(model.H(use_cells_double,:),cur_area);
  hold on; scatter(cur_model_H(:,2), cur_model_H(:,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
  if do_figure_exports, f_set_axis_props(gca); f_export_fig(['fig_ch2_data_01_results_image_zoom_' num2str(i1) '_morecells']); end
  
  
  % put border on the original images
  %figure(4); hold on; rectangle('Position',get_area_rect(cur_area), 'EdgeColor',cur_border_color,'Linewidth',1.);
  figure(5); hold on; rectangle('Position',get_area_rect(cur_area), 'EdgeColor',cur_border_color,'Linewidth',1.);
end


%% Do the variance plots
% Run get_reconst_example_1 on the cluster to get this array
load(['~/Data/CHOMP_Examples/reconst_example_' chomp_timestamp '_iter_' chomp_iter '.mat'])



%%

% Show the variance image
inds = get_area_inds(cur_area);

mom1 = 2;
my_new_figure_image(100+mom1*10+1); clf; imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})); axis image; colorbar; colormap(my_colormaps.fire)
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_data_01_example_variance_zoom'); end

% Get a single line of the variance and thus covariance
patch_mask = zeros(szPatch(1),szPatch(2));
patch_mask(:,45) = 1;

% Add the patch-mask as a line of black onto the image;
hold on; line([45,45],[1,49], 'LineWidth', 2., 'Color', 'black' )
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_data_01_example_variance_zoom_line'); end



full_var = opt.cumulant_pixelwise{2}(inds{1},inds{2});
line_var = full_var(logical(patch_mask(:)));
my_new_figure_image(101); clf;
plot(line_var, ...
  'LineWidth', 3., 'Color', [0 0 0]...
  )
ylabel('Variance (a.u.)')
xlabel('Y-Location (pixel)')
f_set_axis_props(gca);
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_data_01_example_line_variance_plot'); end

patch_mask_outer = patch_mask(:)*patch_mask(:)';
line_covar = data_cov(logical(patch_mask_outer(:)));
%figure; imagesc(reshape(line_covar,sum(patch_mask(:)),sum(patch_mask(:))));

clim_covar = [min(line_covar), 6e6];
my_new_figure_image(102); imagesc(reshape(line_covar,sum(patch_mask(:)),sum(patch_mask(:))), clim_covar); colormap(my_colormaps.fire);
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_data_01_example_line_covariance_image'); end


%% Box covariance


mom1 = 2;
my_new_figure_image(100+mom1*10+1); clf; imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})); axis image; colorbar; colormap(my_colormaps.fire)
%if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_data_01_example_variance_zoom'); end


% Get a box of the variance and thus covariance
patch_mask = zeros(szPatch(1),szPatch(2));
patch_mask(5:25,45:60) = 1;

% Add the patch-mask as a line of black onto the image;
hold on; rectangle('Position', [45,5, 15,20], 'LineWidth', 2., 'EdgeColor', 'black' )
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_data_01_example_variance_zoom_box'); end


patch_mask_outer = patch_mask(:)*patch_mask(:)';
box_covar = data_cov(logical(patch_mask_outer(:)));
my_new_figure_image(103); imagesc(reshape(box_covar,sum(patch_mask(:)),sum(patch_mask(:))), clim_covar); colormap(my_colormaps.fire);
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_data_01_example_box_covariance_image'); end


box_covar_reconst = full_reconst_covmat(logical(patch_mask_outer(:)));
box_covar_resid = box_covar - box_covar_reconst;



%% Full covariance

reconst_type = '_offdiag_only';
clim_covar = [0e6, 6e6];
%my_new_figure_image(200); imagesc(data_cov, clim_covar); colormap(my_colormaps.fire);
my_new_figure_image(200); clf; imagesc(gauss_smooth_matrix(data_cov,1.2,[50,99,148],1.), clim_covar); colormap(my_colormaps.fire); colorbar; axis image

%if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster('fig_ch2_data_01_example_line_covariance_image'); end

   % Show full covariance reconstruction as well
  clim = clim_covar;
  my_new_figure_image(201); imagesc(gauss_smooth_matrix(data_cov,1.2,[50,99,148],1.), clim); axis off; axis image; colorbar; colormap(my_colormaps.fire)
  if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster(['fig_ch2_data_01_example_mom' num2str(mom1) '_full_orig']); end

  my_new_figure_image(202); imagesc(gauss_smooth_matrix(full_reconst_covmat,1.2,[50,99,148],1.), clim); axis off; axis image; colorbar; colormap(my_colormaps.fire)
  if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster(['fig_ch2_data_01_example_mom' num2str(mom1) '_full_reconst', reconst_type]); end

  my_new_figure_image(203); imagesc(gauss_smooth_matrix(data_cov - full_reconst_covmat,1.2,[50,99,148],1.), clim);; axis off; axis image; colorbar; colormap(my_colormaps.fire)
  if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster(['fig_ch2_data_01_example_mom' num2str(mom1) '_full_resid', reconst_type]); end


%%

 %inds = get_area_inds(cur_area);
%   for mom1 = 2:opt.mom
%   
%     mom_mode = 0; %mom_mode = get_hist_mode(opt.cumulant_pixelwise{mom1},500);
%     clim = [min(min(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})-mom_mode)), max(max(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})-mom_mode))];
%     %clim(1) = 0.
%     
%     % Show plots of various diagonal reconstructions
%     
%    
% %     my_new_figure_image(100+mom1*10+1); imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})-mom_mode ,clim); axis off; axis image; colorbar; colormap(my_colormaps.fire)
% %     if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster(['fig_ch2_data_01_example_mom' num2str(mom1) '_diag_orig']); end
% % 
% %     
% %     my_new_figure_image(100+mom1*10+2); imagesc(full_reconst_diag{mom1}(inds{1},inds{2}),clim); axis off; axis image; colorbar; colormap(my_colormaps.fire)
% %     if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster(['fig_ch2_data_01_example_mom' num2str(mom1) '_diag_reconst' reconst_type]); end
% % 
% %     my_new_figure_image(100+mom1*10+3); imagesc(opt.cumulant_pixelwise{mom1}(inds{1},inds{2})-mom_mode-(full_reconst_diag{mom1}(inds{1},inds{2})),clim); axis off; axis image; colorbar; colormap(my_colormaps.fire)
% %     if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster(['fig_ch2_data_01_example_mom' num2str(mom1) '_diag_resid', reconst_type]); end
% 
%   end
  
%      % Show full covariance reconstruction as well
%   mom_mode = 0; % mom_mode = get_hist_mode(opt.cumulant_pixelwise{mom1},500);
%   clim = clim_covar;
%   my_new_figure_image(201); imagesc(data_cov - mom_mode*eye(size(data_cov,1)),clim); axis off; axis image; colorbar; colormap(my_colormaps.fire)
%   if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster(['fig_ch2_data_01_example_mom' num2str(mom1) '_full_orig']); end
% 
%   my_new_figure_image(202); imagesc(full_reconst_covmat,clim); axis off; axis image; colorbar; colormap(my_colormaps.fire)
%   if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster(['fig_ch2_data_01_example_mom' num2str(mom1) '_full_reconst', reconst_type]); end
% 
%   my_new_figure_image(203); imagesc(data_cov - mom_mode*eye(size(data_cov,1)) - full_reconst_covmat,clim); axis off; axis image; colorbar; colormap(my_colormaps.fire)
%   if do_figure_exports,f_set_axis_props(gca); f_export_fig_raster(['fig_ch2_data_01_example_mom' num2str(mom1) '_full_resid', reconst_type]); end



%% Plot the precision-recall curve


% if exist(f_getfigfullpath('neurofinder_results_curve'),'file')
%   load(f_getfigfullpath('neurofinder_results_curve'));
% else
%   [~, ROIs] = getROIs(opt, use_cells_double);
%   num_cells_iters = 10:10:max(use_cells_double);
%   cur_neurofinder_results = get_neurofinder_results( opt, ROIs, 1, num_cells_iters);
% 
%   save(f_getfigfullpath('neurofinder_results_curve'), 'cur_neurofinder_results');
% end
% 
% my_orange = [1.0000    0.4980    0.0549];
% my_new_figure_image(6); gcf
% plot(...
%   cur_neurofinder_results(:,1),cur_neurofinder_results(:,[2]), ...
%   cur_neurofinder_results(:,1),cur_neurofinder_results(:,[4]), ...
%   'LineWidth', 3., 'Color', {[0 0 0], my_orange}...
%   )
% legend({'Recall','Precision'})
% f_set_axis_props(gca);
% 
% %hold on; plot(orig_H(:,2), orig_H(:,1), 'or' , 'Linewidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'r', 'MarkerEdgeColor','r');
% 
