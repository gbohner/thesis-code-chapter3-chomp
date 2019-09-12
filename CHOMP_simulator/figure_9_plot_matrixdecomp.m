%Choose timestamp
clear
% %old runs
% all_ts{1} = '20160429T052032'; %single_dist = 1; % Single source distribution in a field of view
% all_ts{2} = '20160429T051708'; %single_dist = 0.5; % WIthin 2 orders of magnitude distributions within a field of view
%all_ts{3} = '20160429T051800'; %single_dist = 0;  % Arbitrary source distribution in a field of view

% %New run
all_ts{1} = '20160512T175749'; %single_dist = 1; % Single source distribution in a field of view
all_ts{2} = '20160512T175852'; %single_dist = 0.5; % WIthin 2 orders of magnitude distributions within a field of view
all_ts{3} = '20160512T175856'; %single_dist = 0;  % Arbitrary source distribution in a field of view

% %Runs that contains matrix decomp data
% %Runs with low MAXITER=300 on lasso (worse localisation performance)
% all_ts{4} = '_matdecomp_dist_10'; %single_dist = 1;  % Arbitrary source distribution in a field of view
% all_ts{5} = '_matdecomp_dist_05';
% all_ts{6} = '_matdecomp_dist_00'; %single_dist = 0; % WIthin 2 orders of magnitude distributions within a field of view

% Runs with default MAXITER=1000 on lasso (better localisation performance)
all_ts{6} = '20160729T002728'; %single_dist = 0; % WIthin 2 orders of magnitude distributions within a field of view
all_ts{4} = '20160729T002642'; %single_dist = 1;  % Arbitrary source distribution in a field of view
all_ts{5} = '20160729T002658'; %single_dist = 0.5; % Single source distribution in a field of view


results_folder = '';

all_auc_means =[];
all_auc_vars =[];
all_auc_counts =[];
figure('Units','normalized','Position',[0,0.1,0.6,0.7],'Color','none');
%set(gcf,'Visible','off')
for i11=1:6
  
    ts = all_ts{i11};
    
    i1 = mod(i11-1,3)+1;

    load(['figure_5_results' ts], 'results*');
    load(['init_params' ts]);
    load(['X_samps' ts],'means','vars','kurs','nsamps');
 

    MOMs = [MOMs, Inf]
    
    results_runparams(results_recalls(:,3)==0,:,:) = [];
    results_recalls(results_recalls(:,3)==0,:) = [];
    if i11 > 3
      results_runtimes(results_recalls(:,3)==0,:) = [];
      all_results_runtimes{i1} = results_runtimes;
    end
    
    f_m = @(selection)([mean(results_recalls(selection,2) - results_recalls(selection,1)), mean(results_recalls(selection,3) - results_recalls(selection,1))]);
    f_v = @(selection)([var(results_recalls(selection,2) - results_recalls(selection,1)), var(results_recalls(selection,3) - results_recalls(selection,1))]);
    f_c = @(selection)(sum(selection));

    %Figure for AUC gain over the mean
    %All sources
    selection = logical(ones(size(results_runparams,1),1));
    auc_means(:,1) = f_m(selection);
    auc_vars(:,1) = f_v(selection);
    auc_counts(:,1) = f_c(selection);


    %High variance runs
    selection = (results_runparams(:,2)>=7); %High variance runs
    auc_means(:,2) = f_m(selection);
    auc_vars(:,2) = f_v(selection);
    auc_counts(:,2) = f_c(selection);

    %Low mean runs
    selection = (results_runparams(:,1)<=2); %Low mean runs
    auc_means(:,3) = f_m(selection);
    auc_vars(:,3) = f_v(selection);
    auc_counts(:,3) = f_c(selection);

    %High kurtosis
    selection = logical((results_runparams(:,3)>=3).*(results_runparams(:,1)<=1).*(results_runparams(:,2)>=10)); %High kurtosis runs
    auc_means(:,4) = f_m(selection);
    auc_vars(:,4) = f_v(selection);
    auc_counts(:,4) = f_c(selection);

    all_auc_counts = [all_auc_counts, auc_counts];
    all_auc_means = [all_auc_means, auc_means];
    all_auc_vars = [all_auc_vars, auc_vars];

    hold on;
    res = results_recalls;
    %sel = selection;
    sel = logical(ones(size(results_runparams,1),1));
    markersize=10;
    K = size(results_recalls(sel,1),1);
    if i11 <=3
      hs1 = scatter(i1*ones(K,1)-0.3+randn(K,1)*0.035,res(sel,1),markersize,[0.7 0.7 0.7],'filled');
      hs2 = scatter(i1*ones(K,1)-0.1+randn(K,1)*0.035,res(sel,2),markersize,[0 0 0.5],'filled');
      hs3 = scatter(i1*ones(K,1)+0.1+randn(K,1)*0.035,res(sel,3),markersize,[0.9 1 0],'filled');
    else
      hs4 = scatter(i1*ones(K,1)+0.3+randn(K,1)*0.05,res(sel,4),markersize,[1 0 0.9],'filled');
    end

    %Plot the quantiles in bigger

    quants = quantile(res(sel,:),normcdf([-2,-1,0,1,2],0,1),1);
    colors = [[0.7 0.7 0.7]; [0 0 0.5]; [0.9 1 0]; [1 0 0.9]];
    opp_color = @(c) [~c(:,1:2) ~c(:,3).*~xor(c(:,1),c(:,2))];
    opp_colors = opp_color(round(colors));
    if i11 <= 3
      scatter(reshape(repmat([i1-0.3,i1-0.1,i1+0.1],1,1),1,[]),quants(5,:)',1*markersize,opp_colors(1:3,:),'^')
      scatter(reshape(repmat([i1-0.3,i1-0.1,i1+0.1],1,1),1,[]),quants(4,:)',3*markersize,opp_colors(1:3,:),'^')
      scatter(reshape(repmat([i1-0.3,i1-0.1,i1+0.1],1,1),1,[]),quants(3,:)',10*markersize,opp_colors(1:3,:),'d','filled')
      scatter(reshape(repmat([i1-0.3,i1-0.1,i1+0.1],1,1),1,[]),quants(2,:)',3*markersize,opp_colors(1:3,:),'v')
      scatter(reshape(repmat([i1-0.3,i1-0.1,i1+0.1],1,1),1,[]),quants(1,:)',1*markersize,opp_colors(1:3,:),'v')
    else
      scatter(reshape(repmat([i1+0.3],1,1),1,[]),quants(5,end)',1*markersize,opp_colors(end,:),'^')
      scatter(reshape(repmat([i1+0.3],1,1),1,[]),quants(4,end)',3*markersize,opp_colors(end,:),'^')
      scatter(reshape(repmat([i1+0.3],1,1),1,[]),quants(3,end)',10*markersize,opp_colors(end,:),'d','filled')
      scatter(reshape(repmat([i1+0.3],1,1),1,[]),quants(2,end)',3*markersize,opp_colors(end,:),'v')
      scatter(reshape(repmat([i1+0.3],1,1),1,[]),quants(1,end)',1*markersize,opp_colors(end,:),'v')
    end
      
end
  

xlim([0.5,3.5])    
set(gca,'XTick',[1,2,3])
set(gca,'XTickLabel',{'No mixing','Realistic mixture','Uniform mixture'})
ylim([-0.003,1])
set(gca,'YTick',[0,0.5,1])
ylabel('Area Under Recall Curve')
hl = legend([hs1,hs2,hs3,hs4],{'R=1','R=2','R=4','Group Lasso'},'Location','NorthEast');
set(gca,'FontSize',16)
set(hl,'FontSize',16)
legend('boxoff')
set(gca,'Position',[0.1,0.1,0.85,0.85]);
%export_fig 'figure_9_new.pdf' '-q101'

%% Figure barchart
% % close all
% % b = bar(all_auc_means(:,1:9)');
% % b(1).FaceColor=[0 0 0.5];
% % b(2).FaceColor=[0.9 1 0];
% % legend({'R=2','R=4'},'FontSize',16)
% % legend('boxoff')
% % ylabel('Recall gain over mean')
% % 
% % hold on;
% % be = errorbar([], [],'LineStyle','none');
% % set(be,'XData',all_auc_means(:,1:9)')
% % set(be,'YData',sqrt(all_auc_vars(:,1:9)'))
% 
% %,sqrt(all_auc_vars(:,1:9)));
% 
% %figure('Units','normalized','Position',[0.1,0.1,0.8,0.8]);
% use_groups = [1,2,3,5,6,7,9]; %1:9; %[5,6,7,8];
% % Bar names 1:4 single_dist, 5:8 close_dists, 9:12 all_dists
% bar_names = {'No mixing   Every run  ','No mixing     High variance    ','No mixing      Low mean     ','No mixing High kurtosis', ...
%   'Realistic mixture Every run','Realistic mixture High variance','Realistic mixture Low mean','Realistic mixture High kurtosis', ...
%   'Uniform mixture Every run'};
% 
% [~,b,eb] = errorbar_groups(all_auc_means(:,use_groups),2*sqrt(all_auc_vars(:,use_groups))./repmat(sqrt(all_auc_counts(:,use_groups)),[2,1]), ...
%   'bar_width',0.75,'errorbar_width',1, ...
%   'bar_names',bar_names(use_groups));
% 
% set(gcf,'Units','normalized','Position',[0,0.1,1,0.7],'Color','none')
% 
% set(gca,'FontSize',16)
% 
% set(b(1),'FaceColor',[0 0 0.5]');
% set(b(2),'FaceColor',[0.9 1 0]');
% legend({'R=2','R=4'},'FontSize',16)
% legend('boxoff')
% ylabel('AURC Increase')
% set(gca,'Position',[0.1,0.1,0.85,0.85]);
% %fix_xticklabels(gca,0.12,{'FontSize',15}); %Use for 9 columns
% fix_xticklabels(gca,0.26,{'FontSize',15}); %Use for 7 columns
% 
% export_fig 'figure_10_new.pdf' '-q101'
%%  b(2).FaceColor=;


%% Figure runtimes

figure('Units','normalized','Position',[0,0.1,0.45,0.8],'Color','none');
for i1 = 1:3
  K = size(all_results_runtimes{i1},1);
  for j1 = 1:4
    scatter(j1*ones(K,1)+randn(K,1)*0.15, all_results_runtimes{i1}(:,j1),markersize,colors(j1,:));
    hold on;
  end
end
set(gca, 'yscale', 'log')
set(gca,'XTick',[1,2,3,4])
set(gca,'XTickLabel',{'R=1','R=2','R=4', 'Group Lasso'})
xlim([0.4, 4.6])
ylabel('Running time (sec)')
set(gca,'FontSize',16)
set(gca,'Position',[0.1,0.1,0.85,0.85]);
export_fig 'figure_11_new.pdf' '-q101'