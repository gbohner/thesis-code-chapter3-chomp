addpath(genpath('.'));

env_home = 0;
dataset_id = '00.00';

% Load
[opt, model] = load_best_fits(dataset_id, env_home);

% Get ROIs
[ROI_mask, ROIs] = getROIs(opt);
[orig_ROIs, orig_H, orig_ROI_mask] = get_neurofinder_orig_ROIs(opt);



% helper funcs to plot zoomed in area
get_cur_H_inds = @(H,cur_area)(H(:,1)>cur_area(1) & H(:,1)<cur_area(3) & H(:,2)>cur_area(2) & H(:,2)<cur_area(4));
get_cur_H = @(H, cur_area)([H(get_cur_H_inds(H,cur_area),1)-cur_area(1),H(get_cur_H_inds(H,cur_area),2)-cur_area(2)]);
get_area_inds = @(cur_area){cur_area(1):cur_area(3), cur_area(2):cur_area(4)};
get_area_rect = @(cur_area)[cur_area(2), cur_area(1), cur_area(4)-cur_area(2), cur_area(3)-cur_area(1)];


if strcmp(dataset_id, '00.00')  
  %rawY_preproc = chomp_data(get_path(opt, 'raw_virtual_stack'));
  % Get non-imputed preproc data
  rawY_preproc = chomp_data('/nfs/data/gergo/Neurofinder_update/neurofinder.00.00/preproc2P/CHOMP/input/neurofinder.00.00_preproc2P_expertPrior_unampLik_gitsha_2bd0d72_evalgit_db4ade8_noimpute_rPC_1_origPMgain_useNans_targetCoverage_10_grid_30_7_virtual_stack_raw.chd');
  rawY_orig = chomp_data('/nfs/data/gergo/Neurofinder_update/neurofinder.00.00/CHOMP/input/neurofinder.00.00_virtual_stack_raw.chd');
  
  % Zoom 1 (area_51, middle)
  area_51 = [285,70,400,185];
  %createNeurofinderVideo(data_path, '.tif', 1:2000, [], get_area_inds(area_51), [], 0)
  
  [timeseries_preproc, raw_mean_preproc, raw_var_preproc] = get_cell_timeseries_neurofinder(rawY_preproc,ROIs(get_cur_H_inds(model.H,area_51)));
  timeseries_preproc_neurofinder = get_cell_timeseries_neurofinder(rawY_preproc,orig_ROIs(get_cur_H_inds(orig_H,area_51)),[],1);
  
  [timeseries_orig, raw_mean_orig, raw_var_orig] = get_cell_timeseries_neurofinder(rawY_orig,ROIs(get_cur_H_inds(model.H,area_51)),1:2000);
  timeseries_orig_neurofinder = get_cell_timeseries_neurofinder(rawY_orig,orig_ROIs(get_cur_H_inds(orig_H,area_51)),1:2000,1);
  
  
  save(['./Examples/timeseries_example_' dataset_id '_area51_' datestr(now,30) '.mat'], ...
      'timeseries_preproc', 'timeseries_preproc_neurofinder', 'raw_mean_preproc', 'raw_var_preproc', ...
    'timeseries_orig', 'timeseries_orig_neurofinder','raw_mean_orig', 'raw_var_orig' );
  

  % Zoom 2 (area_52, top right)
  area_52 = [5,391,121,506];
  %createNeurofinderVideo(data_path, '.tif', 1:2000, [], get_area_inds(area_52), [], 0)
  [timeseries_preproc, raw_mean_preproc, raw_var_preproc] = get_cell_timeseries_neurofinder(rawY_preproc,ROIs(get_cur_H_inds(model.H,area_52)));
  timeseries_preproc_neurofinder = get_cell_timeseries_neurofinder(rawY_preproc,orig_ROIs(get_cur_H_inds(orig_H,area_52)),[],1);
  
  [timeseries_orig, raw_mean_orig, raw_var_orig] = get_cell_timeseries_neurofinder(rawY_orig,ROIs(get_cur_H_inds(model.H,area_52)),1:2000);
  timeseries_orig_neurofinder = get_cell_timeseries_neurofinder(rawY_orig,orig_ROIs(get_cur_H_inds(orig_H,area_52)),1:2000,1);
  
  
  save(['./Examples/timeseries_example_' dataset_id '_area52_' datestr(now,30) '.mat'], ...
      'timeseries_preproc', 'timeseries_preproc_neurofinder', 'raw_mean_preproc', 'raw_var_preproc', ...
    'timeseries_orig', 'timeseries_orig_neurofinder','raw_mean_orig', 'raw_var_orig' );
  
end

if strcmp(dataset_id, '01.00')
  
end


if strcmp(dataset_id, '02.00')  

  % 20190529T060454, iter 5 -> '20190607T163423'
  save(['./Examples/timeseries_example_' dataset_id '_full_' '20190607T163423' '.mat'], ...
      'timeseries','timeseries_neurofinder', 'raw_mean', 'raw_var');
end

%% Plotting

plot_these_series = {timeseries_preproc, timeseries_orig, timeseries_preproc_neurofinder, timeseries_orig_neurofinder};
use_these_stats = {[raw_mean_preproc, raw_var_preproc], [raw_mean_orig, raw_var_orig], [raw_mean_preproc, raw_var_preproc], [raw_mean_orig, raw_var_orig]};

for j1 = 1:length(plot_these_series)
  cur_timeseries = plot_these_series{j1};
  cur_raw_mean = use_these_stats{j1}(1);
  cur_raw_var = use_these_stats{j1}(2);
  
  my_new_figure_image(1840 + j1); clf;
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
end

% update_visualize_model(model, use_cells, 0);
% hold on;
% hold on; plot(orig_H(to_plot,2), orig_H(to_plot,1), 'or' , 'Linewidth', 2, 'MarkerSize', 2, 'MarkerFaceColor', 'b', 'MarkerEdgeColor','b');
% for i12 = to_plot%size(cur_orig_H,1)
%   text(orig_H(i12,2), orig_H(i12,1), num2str(i12), 'Color', 'b','FontSize',20,'FontWeight','bold');
% end