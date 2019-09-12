do_figure_exports = 1;
if do_figure_exports
  clear all;
  do_figure_exports = 1;
  addpath('export_fig-master/');
  %fig_timestamp = ['_' datestr(now,30)];
  fig_timestamp = '_20190530T132106';
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
orig_model_00 = load('/mnt/gatsby/nfs/data/gergo/Neurofinder_update/neurofinder.00.00/CHOMP/output/neurofinder.00.00_20170727T140331_iter_3.mat');


% preProcessing params
prior = 'expertPrior';%prior = 'noPrior';
lik = 'unampLik'; %lik = 'linLik';
gitsha = 'gitsha_2bd0d72_evalgit_db4ade8';
trainType = '_rPC_1_origPMgain_useNans';
targetCoverage = '_targetCoverage_10';
gridType = '_grid_30_7';

% Load preprocessed model fit
dataset_name = 'neurofinder.00.00';
chomp_iter = '2';
chomp_timestamp = '20190527T202351';

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

% my_new_figure_image(3); clf; imagesc(orig_model_00.model.y_orig); axis image; axis off; colorbar; colormap gray
% hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
% if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_data_00_orig_image_gray'); end
% 
% 
my_new_figure_image(3); clf; imagesc(orig_model_00.model.y_orig); axis image; axis off; colorbar; 
%hold on; scatter(model.H(use_cells,2), model.H(use_cells,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
%hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
%hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'MarkerEdgeColor','w','Linewidth',0.5,'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
if do_figure_exports, f_set_axis_props(gca); f_export_fig('fig_ch2_data_00_orig_image_green'); end

my_new_figure_image(4); clf; imagesc(opt.cumulant_pixelwise{1}); axis image; axis off; colorbar; 
if do_figure_exports, figure(4); f_set_axis_props(gca); f_export_fig('fig_ch2_data_00_results_image_whiten'); end
hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
if do_figure_exports, figure(4); f_set_axis_props(gca); f_export_fig('fig_ch2_data_00_results_image_whiten_origmarks'); end
hold on; scatter(model.H(use_cells,2), model.H(use_cells,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
if do_figure_exports, figure(4); f_set_axis_props(gca); f_export_fig('fig_ch2_data_00_results_image_whiten_allmarks'); end


my_new_figure_image(5); clf; imagesc(model.y_orig); axis image; axis off; colorbar; 
if do_figure_exports, figure(5); f_set_axis_props(gca); f_export_fig('fig_ch2_data_00_results_image_main'); end
hold on; scatter(orig_H(:,2), orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
if do_figure_exports, figure(5); f_set_axis_props(gca); f_export_fig('fig_ch2_data_00_results_image_main_origmarks'); end
hold on; scatter(model.H(use_cells,2), model.H(use_cells,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
if do_figure_exports, figure(5); f_set_axis_props(gca); f_export_fig('fig_ch2_data_00_results_image_main_allmarks'); end




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
  
  if do_figure_exports, f_set_axis_props(gca); f_export_fig(['fig_ch2_data_00_results_image_zoom_' num2str(i1)]); end
  
  % Show what if we added more cells
  cur_model_H = get_cur_H(model.H(use_cells_double,:),cur_area);
  hold on; scatter(cur_model_H(:,2), cur_model_H(:,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
  if do_figure_exports, f_set_axis_props(gca); f_export_fig(['fig_ch2_data_00_results_image_zoom_' num2str(i1) '_morecells']); end
  
  % put border on the original images
  %figure(4); hold on; rectangle('Position',get_area_rect(cur_area), 'EdgeColor',cur_border_color,'Linewidth',1.);
  figure(5); hold on; rectangle('Position',get_area_rect(cur_area), 'EdgeColor',cur_border_color,'Linewidth',1.);
end

if do_figure_exports, figure(5); f_set_axis_props(gca); f_export_fig('fig_ch2_data_00_results_image_main_allmarks_zooms'); end

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
if do_figure_exports, figure(5); f_set_axis_props(gca); f_export_fig('fig_ch2_data_00_results_image_main_allmarks_zooms_morecells'); end


%% Plot timeseries belonging to the zoomed in areas - based on CHOMP

struct2workspace=@(mystruct)cellfun(@(x,y) assignin('base',x,y),fieldnames(mystruct),struct2cell(mystruct));

struct_timeseries_area51 = load('/Users/gergobohner/Data/CHOMP_Examples/timeseries_example_00.00_area51_20190608T093331.mat');
struct_timeseries_area52 = load('/Users/gergobohner/Data/CHOMP_Examples/timeseries_example_00.00_area52_20190608T093552.mat'); % Area 52

all_struct_timeseries = {struct_timeseries_area51, struct_timeseries_area52};

% Do the timeseries
for i111 = 1:length(all_struct_timeseries)

  struct2workspace(all_struct_timeseries{i111}); % Load to workspace
  max_cells = 10;

  plot_these_series = {timeseries_preproc, timeseries_orig, timeseries_preproc_neurofinder, timeseries_orig_neurofinder};
  use_these_stats = {[raw_mean_preproc, raw_var_preproc], [raw_mean_orig, raw_var_orig], [raw_mean_preproc, raw_var_preproc], [raw_mean_orig, raw_var_orig]};
  use_these_plotnames = {'preproc_CHOMPtop10','orig_CHOMPtop10'};
  
  for j1 = 1:2%length(plot_these_series)
    cur_timeseries = plot_these_series{j1};
    cur_raw_mean = use_these_stats{j1}(1);
    cur_raw_var = use_these_stats{j1}(2);

    my_new_figure_image(1840 + i111*10 + j1); clf;
    % cur_timeseries = timeseries_neurofinder;
    % figure(1843); clf;
    %v = max(std(cur_timeseries(to_plot,:),1))*2;
    v = 3; yticks = []; yticklabels = {};

    %cur_timeseries_zscored = reshape(zscore(cur_timeseries(:)),size(cur_timeseries)); % jointly zscore for clarity
    cur_timeseries_zscored = (cur_timeseries-cur_raw_mean)./sqrt(cur_raw_var); % zscore based on all data stats

    to_plot = [1:min(size(cur_timeseries,1),10)];%[15:20];%[10:15]+30;
    %to_plot = [86,94,97,100,117,134, 89,72,87];
    if j1>=3 % neurofinder rois - reorder them
      [~, to_plot] = sort(sum(cur_timeseries_zscored.^2,2),'descend');
      to_plot = to_plot(1:min(size(cur_timeseries,1),10));
    end
    %to_plot = to_plot(end-20:end);

    all_colors = mat2cell(my_colormaps.felfire(round(linspace(150,1,numel(to_plot))),:),ones(numel(to_plot),1));
    for i1 = 1:length(to_plot)
      plot(cur_timeseries_zscored(to_plot(i1),:) + numel(to_plot)*v - i1*v, 'LineWidth', 2, 'Color', all_colors{i1}); hold on;
      line([1,size(cur_timeseries,2)], [numel(to_plot)*v - i1*v,numel(to_plot)*v - i1*v], 'LineStyle',':','Color','k');
      yticks = [yticks, numel(to_plot)*v - i1*v];
      yticklabels{end+1} = sprintf('Cell %3d',to_plot(i1));
      xlabel('Frame')
    end
    set(gca,'YTick',yticks(end:-1:1))
    set(gca,'YTickLabel',yticklabels(end:-1:1))
    ylim([-2*v, (2+numel(to_plot))*v]);
    % Add v-height bar
    line([1500,1500], [(-1.5)*v, (-0.5)*v], 'LineWidth',6,'Color','k');
    % Add legend for dotted line
    line([1500,1650], [(-1.6)*v, (-1.6)*v], 'LineStyle',':','Color','k');
    text(1675,(-1.5)*v, 'Dataset mean', 'FontSize',14)
    text(1550,(-0.8)*v, sprintf('%d * Dataset std',v), 'FontSize',14)
    if do_figure_exports, f_set_axis_props(gca); f_export_fig(['fig_ch2_data_00_results_image_zoom_' num2str(i111) '_timeseries_' use_these_plotnames{j1}]); end

  end
  
  
  % Do the corresponding cell numbers on the zoomed images
  % Redo the zoomed images with numbers and only to_plot model dots
  cur_area = cur_zoomed_areas{i111}; cur_border_color = cur_border_colors{i111};
  cur_area_inds = get_area_inds(cur_area);
  cur_model_H = get_cur_H(model.H,cur_area);
  cur_orig_H = get_cur_H(orig_H,cur_area);
  %
  my_new_figure_image(50+i111); clf; imagesc(model.y_orig(cur_area_inds{1},cur_area_inds{2})); axis image; axis off; colorbar
  hold on; scatter(cur_model_H(to_plot,2), cur_model_H(to_plot,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
  hold on; scatter(cur_orig_H(:,2), cur_orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
  hold on; rectangle('Position',[0,0,length(cur_area_inds{1})+1,length(cur_area_inds{2})+1], 'EdgeColor',cur_border_color,'Linewidth',10.);
  hold on; rectangle('Position',[-2,-2,length(cur_area_inds{1})+5,length(cur_area_inds{2})+5], 'EdgeColor','k','Linewidth',1.);

  % Add numbers
  for i12 = 1:length(to_plot)%min(size(cur_model_H,1),10)
    text(cur_model_H(to_plot(i12),2)+2, cur_model_H(to_plot(i12),1)-1, num2str(i12), 'Color', marker_color_model,'FontSize',20,'FontWeight','bold');
  end
  if do_figure_exports, f_set_axis_props(gca); f_export_fig(['fig_ch2_data_00_results_image_zoom_' num2str(i111) '_CHOMPtop10']); end
end



%% Plot timeseries belonging to the zoomed in areas - based on NF ROIs

struct2workspace=@(mystruct)cellfun(@(x,y) assignin('base',x,y),fieldnames(mystruct),struct2cell(mystruct));

struct_timeseries_area51 = load('/Users/gergobohner/Data/CHOMP_Examples/timeseries_example_00.00_area51_20190608T093331.mat');

all_struct_timeseries = {struct_timeseries_area51, struct_timeseries_area52};

% Do the timeseries
for i111 = 1

  struct2workspace(all_struct_timeseries{i111}); % Load to workspace
  max_cells = 10;

  plot_these_series = {timeseries_preproc, timeseries_orig, timeseries_preproc_neurofinder, timeseries_orig_neurofinder};
  use_these_stats = {[raw_mean_preproc, raw_var_preproc], [raw_mean_orig, raw_var_orig], [raw_mean_preproc, raw_var_preproc], [raw_mean_orig, raw_var_orig]};
  use_these_plotnames = {'preproc_CHOMPtop10','orig_CHOMPtop10','preproc_NFmissed'};
  
  % Set to-plot to "missed" NF ROIs
   to_plot = [...
    8,11,12,...
    2,22,24,...
    1,4,19,...
    15, 26]; % simply inactive
  
  missed_by_chomp = [11,12,22,24,19,15,26];
  
  
  for j1 = 3%:2%length(plot_these_series)
    cur_timeseries = plot_these_series{j1};
    cur_raw_mean = use_these_stats{j1}(1);
    cur_raw_var = use_these_stats{j1}(2);

    my_new_figure_image(1840 + i111*10 + j1); clf;
    % cur_timeseries = timeseries_neurofinder;
    % figure(1843); clf;
    %v = max(std(cur_timeseries(to_plot,:),1))*2;
    v = 3; yticks = []; yticklabels = {};

    %cur_timeseries_zscored = reshape(zscore(cur_timeseries(:)),size(cur_timeseries)); % jointly zscore for clarity
    cur_timeseries_zscored = (cur_timeseries-cur_raw_mean)./sqrt(cur_raw_var); % zscore based on all data stats


    all_colors = mat2cell(my_colormaps.felfire(round(linspace(150,1,numel(to_plot))),:),ones(numel(to_plot),1));
    for i1 = 1:length(to_plot)
      plot(cur_timeseries_zscored(to_plot(i1),:) + numel(to_plot)*v - i1*v, 'LineWidth', 2, 'Color', all_colors{i1}); hold on;
      line([1,size(cur_timeseries,2)], [numel(to_plot)*v - i1*v,numel(to_plot)*v - i1*v], 'LineStyle',':','Color','k');
      yticks = [yticks, numel(to_plot)*v - i1*v];
      yticklabels{end+1} = sprintf('Cell %3d',to_plot(i1));
      xlabel('Frame')
      % Add red star for missed cells
      if ismember(to_plot(i1), missed_by_chomp)
        scatter(50, numel(to_plot)*v - i1*v+0.3, 30, 'r*')
      end
    end
    set(gca,'YTick',yticks(end:-1:1))
    set(gca,'YTickLabel',yticklabels(end:-1:1))
    ylim([-2*v, (2+numel(to_plot))*v]);
    % Add v-height bar
    line([1500,1500], [(-1.5)*v, (-0.5)*v], 'LineWidth',6,'Color','k');
    % Add legend for dotted line
    line([1500,1650], [(-1.6)*v, (-1.6)*v], 'LineStyle',':','Color','k');
    text(1675,(-1.5)*v, 'Dataset mean', 'FontSize',14)
    text(1550,(-0.8)*v, sprintf('%d * Dataset std',v), 'FontSize',14)
    if do_figure_exports, f_set_axis_props(gca); f_export_fig(['fig_ch2_data_00_results_image_zoom_' num2str(i111) '_timeseries_' use_these_plotnames{j1}]); end

  end
  
  
  % Do the corresponding cell numbers on the zoomed images
  % Redo the zoomed images with numbers and only to_plot model dots
  cur_area = cur_zoomed_areas{i111}; cur_border_color = cur_border_colors{i111};
  cur_area_inds = get_area_inds(cur_area);
  cur_model_H = get_cur_H(model.H(use_cells_double,:),cur_area);
  cur_orig_H = get_cur_H(orig_H,cur_area);
  %
  my_new_figure_image(50+i111); clf; imagesc(model.y_orig(cur_area_inds{1},cur_area_inds{2})); axis image; axis off; colorbar
  hold on; scatter(cur_model_H(:,2), cur_model_H(:,1), marker_size_model, 'ok', 'Linewidth', marker_linewidth_model, 'MarkerEdgeAlpha',marker_alpha_model,  'MarkerEdgeColor', marker_color_model, 'MarkerFaceColor','none');
  hold on; scatter(cur_orig_H(:,2), cur_orig_H(:,1), marker_size_orig, 'o', 'filled', 'MarkerFaceColor', marker_color_orig, 'MarkerFaceAlpha',marker_alpha_orig);
  hold on; rectangle('Position',[0,0,length(cur_area_inds{1})+1,length(cur_area_inds{2})+1], 'EdgeColor',cur_border_color,'Linewidth',10.);
  hold on; rectangle('Position',[-2,-2,length(cur_area_inds{1})+5,length(cur_area_inds{2})+5], 'EdgeColor','k','Linewidth',1.);

  % Add numbers
  %to_plot = 1:size(cur_orig_H,1);    
  for i12 = 1:length(to_plot)%min(size(cur_model_H,1),10)
    text(cur_orig_H(to_plot(i12),2)+2, cur_orig_H(to_plot(i12),1)-1, num2str(to_plot(i12)), 'Color', marker_color_orig,'FontSize',20,'FontWeight','bold');
  end
  if do_figure_exports, f_set_axis_props(gca); f_export_fig(['fig_ch2_data_00_results_image_zoom_' num2str(i111) '_NFmissed']); end
end



%% Plot the precision-recall curve


if exist(f_getfigfullpath('neurofinder_results_curve_data_0000'),'file')
  load(f_getfigfullpath('neurofinder_results_curve_data_0000'));
else
  [~, ROIs] = getROIs(opt, use_cells_double);
  num_cells_iters = 10:10:max(use_cells_double);
  cur_neurofinder_results = get_neurofinder_results( opt, ROIs, 1, num_cells_iters);

  save(f_getfigfullpath('neurofinder_results_curve_data_0000'), 'cur_neurofinder_results');
end

my_orange = [1.0000    0.4980    0.0549];
my_new_figure_image(6); gcf
cur_h = plot(...
  cur_neurofinder_results(:,1),cur_neurofinder_results(:,[2,4]), ...
  'LineWidth', 3. ...
  );
set(cur_h, {'color'},{[0,0,0]; my_orange})
hold on;
 plot(...
  cur_neurofinder_results(:,1),cur_neurofinder_results(:,[3]),'k--', ...
  'LineWidth', 1. ...
  );
legend({'Recall','Precision','Combined'})
xlabel('Number of cells in the solution set')
if do_figure_exports,f_set_axis_props(gca); f_export_fig('fig_ch2_data_00_results_curve'); end


%hold on; plot(orig_H(:,2), orig_H(:,1), 'or' , 'Linewidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'r', 'MarkerEdgeColor','r');

