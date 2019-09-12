%Choose timestamp
clear
addpath(genpath('.'));
%Old run
ts = '20160729T002728'; %single_dist = 0; % WIthin 2 orders of magnitude distributions within a field of view
%ts = '20160729T002642'; %single_dist = 1;  % Arbitrary source distribution in a field of view
%ts = '20160729T002658'; %single_dist = 0.5; % Single source distribution in a field of view

%New run
%ts = '_matdecomp_dist_00';
ts = '_matdecomp_dist_05';
%ts = '_matdecomp_dist_10';

results_folder = '';

load(['figure_5_results' ts], 'results*');
load(['init_params' ts]);
load(['X_samps' ts],'means','vars','kurs','nsamps');

MOMs = [MOMs, Inf];


% Figure 5
figure('Units','normalized','Position',[0,0.1,0.6,0.7],'Color','none');
smoothing = 1;
for k1 = 1:numel(kurs)
  for mom1 = 1:numel(MOMs)
    subaxis(numel(kurs),numel(MOMs),(numel(kurs)-k1)*numel(MOMs)+mom1, 'Spacing', 0.03, 'MarginLeft',0.2,'MarginBottom',0.2);
    imagesc(imresize((results_found(:,:,k1,mom1)./results_count(:,:,k1))',smoothing), [0,1]);
    %colormap('jet')
    %imagesc(results_count(:,:,k1)', [0.5,1.5]*num_obj*runs/numel(results_count))
    set(gca,'YDir','normal')
    set(gca,'XTick',smoothing:smoothing:(numel(means)*smoothing));
    set(gca,'XTickLabel',means);
    set(gca,'YTick',[1*smoothing, 4*smoothing, 7*smoothing, 10*smoothing]-floor(smoothing/2));
    set(gca,'YTickLabel',vars([1,4,7,10]));
    if k1 == 1 && mom1==1
      xlabel('Signal mean');
      ylabel('Signal variance');
    end
    if k1~=1, set(gca,'XTick',[]), end
    if mom1~=1, set(gca,'YTick',[]), end
    set(gca,'FontSize',14)
  end
end

[ax_supx, h_supx] = suplabel('Cumulants used');
%h_supy = suplabel('Signal kurtosis','y');  
axes(ax_supx); set(gca,'Position',[0.1,0.1,0.85,0.85]); colorbar('Position',[0.93,0.25,0.02,0.6])
set(ax_supx.XAxis,'Visible','on')
set(ax_supx,'XTick',[0.25,0.53,0.82])
set(ax_supx,'XTickLabel',[1,2,4])
set(get(ax_supx,'XLabel'),'Position',get(get(ax_supx,'XLabel'),'Position') + [0.03, -0.01, 0])

set(ax_supx.YAxis,'Visible','on')
set(ax_supx,'YTick',[0.25,0.53,0.82])
set(ax_supx,'YTickLabel',kurs-3)
set(get(ax_supx,'YLabel'),'String','Signal excess kurtosis')
set(get(ax_supx,'YLabel'),'Position',get(get(ax_supx,'YLabel'),'Position') + [-0.01, 0.03, 0])
set(gca,'FontSize',16)
export_fig figure_8_new.pdf -q101