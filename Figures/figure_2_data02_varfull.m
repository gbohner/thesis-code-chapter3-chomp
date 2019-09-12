do_figure_exports = 1;
if do_figure_exports
  clear all;
  do_figure_exports = 1;
  addpath('export_fig-master/');
  %fig_timestamp = ['_' datestr(now,30)];
  fig_timestamp = '_20190605T171329';
  %fig_timestamp = '_20190530T132106';
  %fig_path = '/Users/gergobohner/Dropbox (Personal)/Gatsby/Thesis/Code/Ch2-Matlab-figures/Saved_figures/';
  fig_path = '/Users/gergobohner/Dropbox (Personal)/Gatsby/Thesis/Text_copy/Figures/Chapter2/Neurofinder_results_CHOMP/';
  f_set_axis_props = @(axis_handle)set(axis_handle,'FontSize',16);
  f_getfigfullpath= @(fname)[fig_path fname fig_timestamp];
  f_export_fig = @(fname)export_fig(f_getfigfullpath(fname), '-pdf', '-nocrop', '-q101');
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
%orig_model_00 = load('/mnt/gatsby/nfs/data/gergo/Neurofinder_update/neurofinder.00.00/CHOMP/output/neurofinder.00.00_20170727T140331_iter_3.mat');


% preProcessing params
prior = 'expertPrior';%prior = 'noPrior';
lik = 'unampLik'; %lik = 'linLik';
gitsha = 'gitsha_2bd0d72_evalgit_db4ade8';
trainType = '_rPC_1_origPMgain_useNans';
targetCoverage = '_targetCoverage_10';
gridType = '_grid_30_7';

% Load preprocessed model fit
dataset_name = 'neurofinder.02.00';
chomp_iter = '3'; % Different iters are with different settings!
chomp_timestamp = '20190529T010717';

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
  
orig_path = fileparts(fileparts(fileparts(fileparts(get_path(opt)))));
orig_regions = loadjson([orig_path filesep 'regions' filesep 'regions.json']);
orig_ROIs = cellfun(@(x)x.coordinates+1, orig_regions, 'UniformOutput', false);


% Get the centers for the training ROIs
orig_ROI_centers = cellfun(@(X)round(mean((X-1)*opt.spatial_scale+1))', orig_ROIs, 'UniformOutput', false);
orig_H = [cell2mat(orig_ROI_centers)' ones(length(orig_ROI_centers),1)];
orig_ROI_mask = zeros([size(model.y_orig,1), size(model.y_orig,2), 3]);
for i1 = 1:size(orig_ROIs,2)
  cur_color = 0.5*rand(1,3);
  try
    for j1 = 1:size(orig_ROIs{i1},1)
      orig_ROI_mask(orig_ROIs{i1}(j1,1), orig_ROIs{i1}(j1,2),:) = cur_color;
    end
  end
end

%%

% Plot settings for markers
marker_size_orig = 15;
marker_size_model = 30;
marker_color_orig = 'b';
marker_color_model = [1.0000    0.4980    0.0549];
marker_alpha_orig = 1.0;
marker_alpha_model= 0.8;
marker_linewidth_model = 1.5; 

% Show original model and images
use_cells = 1:size(orig_H,1);
use_cells_double = 1:(2*size(orig_H,1));

my_new_figure_image(3); clf; imagesc(orig_model_00.model.y_orig); axis image; axis off; colorbar; colormap gray
hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_data_02_orig_image_gray'); end


my_new_figure_image(4); clf; imagesc(orig_model_00.model.y_orig); axis image; axis off; colorbar; 
%hold on; scatter(model.H(use_cells,2), model.H(use_cells,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
%hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'MarkerEdgeColor','w','Linewidth',0.5,'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_data_02_orig_image_green'); end



my_new_figure_image(5); clf; imagesc(model.y_orig); axis image; axis off; colorbar; 
hold on; scatter(model.H(use_cells,2), model.H(use_cells,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);

% Zoomed in areas
marker_size_orig = 30;
marker_size_model = 50;
marker_linewidth_model = 2.5;
marker_alpha_model = 1.0;


% helper funcs to plot zoomed in area
get_cur_H_inds = @(H,cur_area)(H(:,1)>cur_area(1) & H(:,1)<cur_area(3) & H(:,2)>cur_area(2) & H(:,2)<cur_area(4));
get_cur_H = @(H, cur_area)([H(get_cur_H_inds(H,cur_area),1)-cur_area(1),H(get_cur_H_inds(H,cur_area),2)-cur_area(2)]);
get_area_inds = @(cur_area){cur_area(1):cur_area(3), cur_area(2):cur_area(4)};
get_area_rect = @(cur_area)[cur_area(2), cur_area(1), cur_area(4)-cur_area(2), cur_area(3)-cur_area(1)];

area_51 = [285,70,400,185];
area_52 = [5,391,121,506];
cur_zoomed_areas = {area_51,area_52};
cur_border_colors = {'b','w'};

for i1 = 1:length(cur_zoomed_areas)
  cur_area = cur_zoomed_areas{i1}; cur_border_color = cur_border_colors{i1};
  cur_area_inds = get_area_inds(cur_area);
  cur_model_H = get_cur_H(model.H(use_cells,:),cur_area);
  cur_orig_H = get_cur_H(orig_H,cur_area);
  my_new_figure_image(50+i1); clf; imagesc(model.y_orig(cur_area_inds{1},cur_area_inds{2})); axis image; axis off; colorbar
  hold on; scatter(cur_model_H(:,2), cur_model_H(:,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
  hold on; scatter(cur_orig_H(:,2), cur_orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
  hold on; rectangle('Position',[0,0,length(cur_area_inds{1})+1,length(cur_area_inds{2})+1], 'EdgeColor',cur_border_color,'Linewidth',10.);
  hold on; rectangle('Position',[-2,-2,length(cur_area_inds{1})+5,length(cur_area_inds{2})+5], 'EdgeColor','k','Linewidth',1.);
  
  if do_figure_exports, f_set_axis_props(gca); f_export_fig(['fig_ch2_data_02_results_image_zoom_' num2str(i1)]); end
  
  % Show what if we added more cells
  cur_model_H = get_cur_H(model.H(use_cells_double,:),cur_area);
  hold on; scatter(cur_model_H(:,2), cur_model_H(:,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
  if do_figure_exports, f_set_axis_props(gca); f_export_fig(['fig_ch2_data_02_results_image_zoom_' num2str(i1) '_morecells']); end
  
  
  % put border on the original images
  %figure(4); hold on; rectangle('Position',get_area_rect(cur_area), 'EdgeColor',cur_border_color,'Linewidth',1.);
  figure(5); hold on; rectangle('Position',get_area_rect(cur_area), 'EdgeColor',cur_border_color,'Linewidth',1.);
end

if do_figure_exports, figure(5); f_set_axis_props(gca); f_export_fig('fig_ch2_data_02_results_image_main'); end

% Add more CHOMP cells to figure 5 too
  % Plot settings for markers
  marker_size_orig = 15;
  marker_size_model = 30;
  marker_color_orig = 'b';
  marker_color_model = [1.0000    0.4980    0.0549];
  marker_alpha_orig = 1.0;
  marker_alpha_model= 0.8;
  marker_linewidth_model = 1.5; 
  
  figure(5);
  hold on; scatter(model.H(use_cells_double,2), model.H(use_cells_double,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
if do_figure_exports, figure(5); f_set_axis_props(gca); f_export_fig('fig_ch2_data_02_results_image_main_morecells'); end


%% Plot the precision-recall curve


if exist(f_getfigfullpath('neurofinder_results_curve'),'file')
  load(f_getfigfullpath('neurofinder_results_curve'));
else
  [~, ROIs] = getROIs(opt, use_cells_double);
  num_cells_iters = 10:10:max(use_cells_double);
  cur_neurofinder_results = get_neurofinder_results( opt, ROIs, 1, num_cells_iters);

  save(f_getfigfullpath('neurofinder_results_curve'), 'cur_neurofinder_results');
end

my_orange = [1.0000    0.4980    0.0549];
my_new_figure_image(6); gcf
plot(...
  cur_neurofinder_results(:,1),cur_neurofinder_results(:,[2]), ...
  cur_neurofinder_results(:,1),cur_neurofinder_results(:,[4]), ...
  'LineWidth', 3., 'Color', {[0 0 0], my_orange}...
  )
legend({'Recall','Precision'})
f_set_axis_props(gca);

%hold on; plot(orig_H(:,2), orig_H(:,1), 'or' , 'Linewidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'r', 'MarkerEdgeColor','r');

