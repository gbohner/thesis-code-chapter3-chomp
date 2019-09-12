%all_dataset_ids = {'00.00','01.00','02.00','03.00','04.00'};
all_dataset_ids = {'04.00'};

%dataset_id = '00.00';

for d11 = 1:length(all_dataset_ids)
  dataset_id = all_dataset_ids{d11};

  max_cell_num = 660;

  [opt, model] = load_best_fits(dataset_id, 1);


  do_figure_exports = 1;
  if do_figure_exports
    addpath('export_fig-master/');
    %fig_timestamp = ['_' datestr(now,30)];
    fig_timestamp = '_20190609T124441';
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


  %% Calculate and do the plot
  if exist([f_getfigfullpath(['neurofinder_results_curve_data_' dataset_id(1:2) dataset_id(4:5) '_' opt.timestamp '_iter_' num2str(opt.niter)]) '.mat'],'file')
    load([f_getfigfullpath(['neurofinder_results_curve_data_' dataset_id(1:2) dataset_id(4:5) '_' opt.timestamp '_iter_' num2str(opt.niter)]) '.mat']);
  else
    [~, ROIs] = getROIs(opt, 1:min(max_cell_num, size(model.H,1)));
    num_cells_iters = 10:10:min(max_cell_num, length(ROIs));
    cur_neurofinder_results = get_neurofinder_results( opt, ROIs, 1, num_cells_iters);

    save(f_getfigfullpath(['neurofinder_results_curve_data_' dataset_id(1:2) dataset_id(4:5) '_' opt.timestamp '_iter_' num2str(opt.niter)]),...
      'cur_neurofinder_results');
  end
  
  %%
  my_orange = [1.0000    0.4980    0.0549];
  my_new_figure_image(6+d11); clf
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
  
  % Also add the "true" number of ROIs according to Neurofinder
  
  [orig_ROIs, orig_H, orig_ROI_mask] = get_neurofinder_orig_ROIs(opt);
  
  if strcmp(dataset_id, '04.00')
    text(300,0.05, sprintf('Neurofinder ROIs: %d', size(orig_H,1)), 'FontSize', 16);
  else
    text(500,0.05, sprintf('Neurofinder ROIs: %d', size(orig_H,1)), 'FontSize', 16);
  end
  
  
  
  if do_figure_exports
    f_set_axis_props(gca); 
    f_export_fig(['neurofinder_results_curve_data_' dataset_id(1:2) dataset_id(4:5) '_' opt.timestamp '_iter_' num2str(opt.niter)]); 
  end
  
end

