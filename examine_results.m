addpath(genpath('Classes'))
addpath(genpath('Subfuncs'))
addpath(genpath('Examples'))
addpath(genpath('Toolboxes'))

%best_model_00 = load('/mnt/gatsby/nfs/data/gergo/Neurofinder_update/neurofinder.00.00/CHOMP/output/neurofinder.00.00_20170727T140331_iter_3.mat');
%best_model_00_00test = load('/mnt/gatsby/nfs/data/gergo/Neurofinder_update/neurofinder.00.00.test/CHOMP/output/neurofinder.00.00.test_20170727T140331_iter_1.mat');
%update_visualize_model(best_model_00.model);
% best_model_01 = load('/mnt/gatsby/nfs/data/gergo/Neurofinder_update/neurofinder.01.00/CHOMP/output/neurofinder.01.00_20170727T144747_iter_3.mat');
% best_model_01_00test = load('/mnt/gatsby/nfs/data/gergo/Neurofinder_update/neurofinder.01.00.test/CHOMP/output/neurofinder.01.00.test_20170727T144747_iter_1.mat');
% update_visualize_model(best_model_01.model)
%best_model_02 = load('/mnt/gatsby/nfs/data/gergo/Neurofinder_update/neurofinder.02.00/CHOMP/output/neurofinder.02.00_20170816T101642_iter_8.mat');
%best_model_02_00test = load('/mnt/gatsby/nfs/data/gergo/Neurofinder_update/neurofinder.02.00.test/CHOMP/output/neurofinder.02.00.test_20170816T101642_iter_1.mat');
% update_visualize_model(best_model_02.model)
% best_model_03 = load('/mnt/gatsby/nfs/data/gergo/Neurofinder_update/neurofinder.03.00/CHOMP/output/neurofinder.03.00_20170727T055501_iter_4.mat');
% update_visualize_model(best_model_03.model, 1:300)
% best_model_04_00 = load('/mnt/gatsby/nfs/data/gergo/Neurofinder_update/neurofinder.04.00/CHOMP/output/neurofinder.04.00_20170727T151012_iter_3.mat');
% update_visualize_model(best_model_04_00.model)
% best_model_04_01 = load('/mnt/gatsby/nfs/data/gergo/Neurofinder_update/neurofinder.04.01/CHOMP/output/neurofinder.04.01_20170727T153355_iter_3.mat');
% update_visualize_model(best_model_04_01.model)


% preProcessing params
prior = 'expertPrior';%prior = 'noPrior';
lik = 'unampLik'; %lik = 'linLik';
%gitsha = 'gitsha_2bd0d720de0995be6b0f1795304839f9877cb6c3';
gitsha = 'gitsha_2bd0d72_evalgit_db4ade8';
trainType = '_rPC_1_origPMgain_useNans';
targetCoverage = '_targetCoverage_10';
gridType = '_grid_30_7';

% CHOMP params

% Big runs
chomp_timestamp ='20190529T060454'; % 2_proper run (var only)
%chomp_timestamp ='20190529T060505'; % 2_proper run (_mean+var only)
%chomp_timestamp ='20190529T010021'; % 2_quick_proper run (var only)
%chomp_timestamp ='20190529T002025'; 
%chomp_timestamp ='20190528T022937'; chomp_iter = '2'; % mom=1 whiten, 
% chomp_timestamp ='20190528T013047'; chomp_iter = '2'; % mom=2 learn basis
% chomp_timestamp ='20190528T025622'; chomp_iter = '3'; % mom=2 learn basis (neurofinder init)
%
dataset_name = 'neurofinder.02.00'; use_cells = 1:250; % testing stuff
chomp_iter = '7';
chomp_timestamp ='20190607T192326'; gitsha = [gitsha '_noimpute']; chomp_iter = '2'; % new try full covar only noimpute
%chomp_timestamp ='20190608T070618'; gitsha = [gitsha '_noimpute']; chomp_iter = '2'; % new try full cokurtosis only noimpute
%chomp_timestamp ='20190608T071927'; gitsha = [gitsha '_noimpute']; chomp_iter = '2'; % full covar only noimpute

% dataset_name = 'neurofinder.02.00'; use_cells = 1:250; % testing stuff
% chomp_iter = '4';

%dataset_00
%dataset_name = 'neurofinder.00.00';
%gridType = '_grid_30_7';%'_grid_50_9';%gridType = '_grid_50_5';
%chomp_iter = '2';
%chomp_timestamp ='20190528T013047';
%chomp_timestamp ='20190527T203118'; % testing local... run, iter 4
% chomp_timestamp ='20190527T202351'; % whitened run, iter 1-2 % seems very good so far, let's see if also on test?! TOO GOOD, 0.80 recall...
%chomp_timestamp ='20190527T202355';
%chomp_timestamp ='20190527T190853';
%chomp_timestamp = '20190527T185944'; % testing one
%chomp_timestamp = '20190527T184232'; % whiten? seems ok...
%chomp_timestamp = '20190527T171644'; % iter 2
% chomp_timestamp = '20190527T170049'; % iter 8
%chomp_timestamp = '20190527T165450';
%chomp_timestamp = '20190526T231702';
%chomp_timestamp = '20190526T231702';
% chomp_timestamp = '20190526T214216';
%chomp_timestamp = '20190526T201325';
%chomp_timestamp = '20190526T193002';
%chomp_timestamp = '20190526T184741';
%chomp_timestamp = '20190526T132821';
%chomp_timestamp = '20190526T130355';
%chomp_timestamp = '20190526T124352';
%chomp_timestamp = '20190526T004154';
% % % %chomp_timestamp = '20190509T032944';
% % % %chomp_timestamp = '20190515T122402';
% % % %chomp_timestamp = '20190515T171732';
% % % %chomp_timestamp = '20190515T172822';
% % % %chomp_timestamp = '20190515T175812';
% % % %chomp_timestamp = '20190515T181314';
% % % %chomp_timestamp = '20190515T213120';
% % % %chomp_timestamp = '20190515T235351';
% % % chomp_timestamp = '20190521T000715';
% % % chomp_timestamp = '20190521T124701';
% % % chomp_timestamp = '20190521T130118';
% % % chomp_timestamp = '20190521T130838';
% % % chomp_timestamp = '20190521T133657';
% % % chomp_timestamp = '20190521T141110';
% % % chomp_timestamp = '20190521T162245';
% % % chomp_timestamp = '20190521T165355';
% % % chomp_timestamp = '20190521T170707';
% % % chomp_timestamp = '20190521T172843';
% % % chomp_timestamp = '20190521T175707';
% % % chomp_timestamp = '20190521T204447'; % this one is 50_9 grid
% % % chomp_timestamp = '20190521T231845';



%dataset 01
% chomp_iter = '6';
% dataset_name = 'neurofinder.01.00';
% chomp_timestamp = '20190528T205007'; % iter 2 currently is very good (even though it was for bugtesting). % iter 3 also awesome!, 4 also
%chomp_timestamp = '20190528T202434'; % after Umap fixing % or not??? very
%bad

%20190527T212938 %tocheck
% chomp_timestamp = '20190527T222253';
%chomp_timestamp = '20190527T212938'; % blank_reconst run, mom=1; seems good!
%chomp_timestamp = '20190526T235419'; % 4th diag order
% %chomp_timestamp = '20190526T190317'; % donut_four simple better
% %chomp_timestamp = '20190526T182426';
% chomp_timestamp = '20190526T182455'; % donut_four simple
%chomp_timestamp = '20190526T133528';
%chomp_timestamp = '20190526T131207';
% % % % % % % % % chomp_timestamp = '20190516T012111';
% % % % % % % % chomp_timestamp = '20190516T143628';
% % % % % % % chomp_timestamp = '20190518T063525';
% % % % % % % chomp_timestamp = '20190518T064545';
% % % % % % % chomp_timestamp = '20190518T225938';
% % % % % % % chomp_timestamp = '20190518T232150';
% % % % % % % chomp_timestamp = '20190520T142125';
% % % % % % % chomp_timestamp = '20190520T152658';
% % % % % % % chomp_timestamp = '20190520T155835';
% % % % % % % chomp_timestamp = '20190520T233735';
% % % % % % chomp_timestamp = '20190520T234819';
% % % % % chomp_timestamp = '20190521T133715';
% % % % % chomp_timestamp = '20190521T141112';
% % % % chomp_timestamp = '20190521T163037';
% % % % chomp_timestamp = '20190521T164743';
% % chomp_timestamp = '20190521T165908';
% % % chomp_timestamp = '20190521T173716'; % Used in reconst examples
% % % chomp_timestamp = '20190521T204038'; % Really long run (1.5 days), offdiagonly

%dataset 02
% dataset_name = 'neurofinder.02.00';
% gridType = '_grid_30_7';
% % chomp_timestamp = '20190516T011804';
% %chomp_timestamp = '20190516T145937';
% %chomp_timestamp = '20190517T140451';
% % chomp_timestamp = '20190517T215337'; %no whiten
% % chomp_timestamp = '20190517T234432'; % with mode subtract and no whiten
% % chomp_timestamp = '20190518T003053';
% % chomp_timestamp = '20190518T013406';
% % chomp_timestamp = '20190518T021335';
% % chomp_timestamp = '20190518T050802';
% % chomp_timestamp = '20190518T060443';
% % chomp_timestamp = '20190518T063516';20190518T064251
% % chomp_timestamp = '20190520T180332';
% % chomp_timestamp = '20190520T195020';
% % chomp_timestamp = '20190520T211948';
% % chomp_timestamp = '20190520T214715';
% % chomp_timestamp = '20190520T223346';
% chomp_timestamp = '20190520T230819';
% chomp_timestamp = '20190522T012238';

%dataset 03
% chomp_iter = '2';
% dataset_name = 'neurofinder.03.00';
% gridType = '_grid_30_7';
% chomp_timestamp = '20190527T225215'; % similar very good whitened run
%chomp_timestamp = '20190527T222028'; % whitened, 1d really good run! 0.75 on train!
% chomp_timestamp = '20190517T234556'; % with mode subtract and no whiten
% chomp_timestamp = '20190518T003100';% with mode subtract and no whiten
% chomp_timestamp = '20190518T010033';% with classic settings
%chomp_timestamp = '20190518T011810';% with classic settings

%dataset 04
% dataset_name = 'neurofinder.04.00';
% gridType = '_grid_30_7';
% % % % chomp_timestamp = '20190518T013749';
% % % % chomp_timestamp = '20190518T014652';
% % % chomp_timestamp = '20190518T021536';
% % chomp_timestamp = '20190520T234018';
% chomp_timestamp = '20190522T014112';


stamp = ['_preproc2P_' prior '_' lik '_' gitsha trainType targetCoverage gridType];

if ~exist('env_home','var')
  env_home = 1;
end


if env_home

  cur_model = load(['/mnt/gatsby/nfs/data/gergo/Neurofinder_update/' ...
    dataset_name '/preproc2P/CHOMP/output/' dataset_name ...
    stamp '_' ...
    chomp_timestamp '_iter_' chomp_iter '.mat']);

  model = cur_model.model;
  model.opt.root_folder = '/mnt/gatsby';
  opt = model.opt;
  
else % on office computer
  cur_model = load(['/nfs/data/gergo/Neurofinder_update/' ...
  dataset_name '/preproc2P/CHOMP/output/' dataset_name ...
  stamp '_' ...
  chomp_timestamp '_iter_' chomp_iter '.mat']);

  model = cur_model.model;
  opt = model.opt;
end

% inp = load(get_path(opt));
% data = inp.data;


%     Copy fro analyze / tmp_visu

[H, W, X, y_orig, y, L] = model.get_fields( 'H', 'W', 'X', 'y_orig','y','L');
szWY = size(model.y);

if ~exist('use_cells', 'var')
  use_cells = 1:size(model.H,1);
end
use_cells = use_cells(1:min([numel(use_cells),size(H,1)]));

update_visualize_model(model, use_cells);

drawnow; pause(1);

% if ~env_home
%   [ROI_mask, ROIs] = getROIs(opt);
%   out = get_neurofinder_results( opt, ROIs, 1 );
% end

%%


%[orig_ROIs, orig_H, orig_ROI_mask] = get_neurofinder_orig_ROIs(opt);

update_visualize_model(model, 1:size(orig_H,1));
figure(3);
colormap(my_colormaps.felfire)
hold on; plot(orig_H(:,2), orig_H(:,1), 'ob' , 'Linewidth', 2, 'MarkerSize', 2, 'MarkerFaceColor', 'b', 'MarkerEdgeColor','b');

%%

% % Set temporary options
% opt_tmp = struct_merge(chomp_options(), opt);
% opt_tmp.niter = chomp_iter;
%       
% [ROI_mask, ROIs] = getROIs(opt_tmp, use_cells);
% 
% % Show original mean image
% figure(2); clf; imagesc(model.y); axis image; colorbar;
% 
% 
% % Get neurofinder results
% try
% orig_path = fileparts(fileparts(fileparts(fileparts(get_path(opt)))));
% orig_regions = loadjson([orig_path filesep 'regions' filesep 'regions.json']);
% orig_ROIs = cellfun(@(x)x.coordinates+1, orig_regions, 'UniformOutput', false);
% 
% 
% % Get the centers for the training ROIs
% orig_ROI_centers = cellfun(@(X)round(mean(X*opt.spatial_scale))', orig_ROIs, 'UniformOutput', false);
% orig_H = [cell2mat(orig_ROI_centers)' ones(length(orig_ROI_centers),1)];
% orig_ROI_mask = zeros([size(ROI_mask,1), size(ROI_mask,2), 3]);
% for i1 = 1:size(orig_ROIs,2)
%   cur_color = 0.5*rand(1,3);
%   try
%     for j1 = 1:size(orig_ROIs{i1},1)
%       orig_ROI_mask(orig_ROIs{i1}(j1,1), orig_ROIs{i1}(j1,2),:) = cur_color;
%     end
%   end
% end
% 
% catch
%   orig_H = H(use_cells,:);
%   orig_ROI_mask = zeros([size(ROI_mask,1), size(ROI_mask,2), 3]);
% end
% 
% %%
% % Show reconstruction of mean
% rh = {};
% rl = {};
% for obj_type = 1:opt.NSS
%   [rh_cur,rl_cur,Wfull_cur] = reconstruct_cell( opt_tmp, W(:,opt.Wblocks{obj_type}), X(use_cells,:) );
%   if obj_type == 1
%     rh = rh_cur;
%     rl = rl_cur;
%   else
%     for mom1 = 1:opt.mom
%       change_cells = use_cells(H(use_cells,3)==obj_type);
%       szRH = size(rh{mom1});
%       rh{mom1} = reshape(rh{mom1},[],szRH(length(szRH)));
%       rh_cur{mom1} = reshape(rh_cur{mom1},[],szRH(length(szRH)));
%       rh{mom1}(:,change_cells) = rh_cur{mom1}(:,change_cells);
%       rh{mom1} = reshape(rh{mom1}, szRH);
%       rl{mom1}(:,:,change_cells) = rl_cur{mom1}(:,:,change_cells);
%     end
%   end
% end
% % opt_tmp.KS = 7;
% % [rh,rl,Wfull] = reconstruct_cell( opt_tmp, W(:,2:end), X(use_cells,2:end) );
% 
% full_reconst = zeros(szWY(1:2));
% for c1 = 1:size(rl{1},3)
%   row_hat = H(c1,1); col_hat = H(c1,2);
%   [inds, cut] = mat_boundary(size(full_reconst),row_hat-floor(opt.m/2):row_hat+floor(opt.m/2),col_hat-floor(opt.m/2):col_hat+floor(opt.m/2));
%   full_reconst(inds{1},inds{2}) = full_reconst(inds{1},inds{2}) + rl{1}(1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2),c1);
% end
% full_reconst_mean = full_reconst;
% 
% figure(4); imagesc(full_reconst_mean); axis image; colorbar;
% figure(411); imagesc(model.y - full_reconst_mean); axis image; colorbar;
% % 
% Show reconstruction of higher moments
mom1 = min(2, opt.mom);
full_reconst = zeros(szWY(1:2));
for c1 = 1:size(rl{mom1},3)
  row_hat = H(c1,1); col_hat = H(c1,2);
  [inds, cut] = mat_boundary(size(full_reconst),row_hat-floor(opt.m/2):row_hat+floor(opt.m/2),col_hat-floor(opt.m/2):col_hat+floor(opt.m/2));
  full_reconst(inds{1},inds{2}) = full_reconst(inds{1},inds{2}) + rl{mom1}(1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2),c1);
end
full_reconst_mom1 = full_reconst;
figure(41); clf; imagesc(full_reconst_mom1); axis image; colorbar;
hold on; plot(orig_H(:,2), orig_H(:,1), 'or' , 'Linewidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'r', 'MarkerEdgeColor','r');
figure(42); clf; imagesc(opt.cumulant_pixelwise{mom1}); axis image; colorbar;
[Zorig, Zmu, Zsig] = zscore(reshape(opt.cumulant_pixelwise{mom1},[],1));

figure(43); clf; imagesc(reshape((reshape(opt.cumulant_pixelwise{mom1}-full_reconst_mom1,[],1)-Zmu)./Zsig,size(opt.cumulant_pixelwise{mom1}))); axis image; colorbar;
hold on; plot(orig_H(:,2), orig_H(:,1), 'or' , 'Linewidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'r', 'MarkerEdgeColor','r');
figure(44); clf; imagesc(reshape(Zorig,size(opt.cumulant_pixelwise{mom1}))); axis image; colorbar;
hold on; plot(orig_H(:,2), orig_H(:,1), 'or' , 'Linewidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'r', 'MarkerEdgeColor','r');

%%
  %figure(43); imagesc(opt.cumulant_pixelwise{mom1} - full_reconst_mom1); axis image; colorbar;

% 
% % Show maximum moment image
% 
% if opt_tmp.mom >= 2
%   load(get_path(opt_tmp)); % Load the input data
%   data = inp.data;
%   % Get the data cumulants
%   data_moments = cell(opt_tmp.mom,1);
%   for mom1 = 1:opt_tmp.mom
%      %data_moments{mom1,1} = mean((inp.data.proc_stack.Y).^mom1,3);
%      data_moments{mom1,1} = mean((data.proc_stack.Y).^mom1,3);
%   end
%   data_cumulants = raw2cum(data_moments);
%   
% %   data_moments_raw = cell(opt_tmp.mom,1);
% %   raw_data = single(inp.data.raw_stack.Y(:,:,:));
% %   for mom1 = 1:opt_tmp.mom
% %      data_moments_raw{mom1,1} = mean(raw_data.^mom1,3);
% %   end
% %   data_cumulants_raw = raw2cum(data_moments_raw);
% %   
% %   % Whiten raw data and get cumulants
% %   opt_tmp2 = opt_tmp;
% %   opt_tmp2.m = round(opt_tmp2.m ./ opt.spatial_scale);
% %   opt_tmp2.smooth_filter_mean = opt_tmp2.smooth_filter_mean ./ opt.spatial_scale;
% %   a_tmp.proc_stack.Y = raw_data;
% %   raw_data_white = whiten_proc(opt_tmp2, a_tmp);
% %   raw_data_white = raw_data_white.proc_stack.Y;
% %   
% %   data_moments_raw_white = cell(opt_tmp.mom,1);
% %   for mom1 = 1:opt_tmp.mom
% %      data_moments_raw_white{mom1,1} = mean(raw_data_white.^mom1,3);
% %   end
% %   data_cumulants_raw_white = raw2cum(data_moments_raw_white);
%   
%   
%   %(alternative, exactly the same - as tested)
%   % W_point = zeros(opt.m);
%   % W_point((opt.m+1)/2, (opt.m+1)/2) = 1;
%   % [data_cumulantsWY, GW, WnormInv] = compute_filters(inp.data, repmat(W_point(:),[1, opt_tmp.KS]), opt_tmp );
%   
%   figure(5); imagesc(data_cumulants{opt_tmp.mom}); axis image; colorbar;

  
  
%   full_reconst_max = zeros(szWY(1:2));
%   for c1 = 1:size(rl{1},3)
%     row_hat = H(c1,1); col_hat = H(c1,2);
%     [inds, cut] = mat_boundary(size(full_reconst_max),row_hat-floor(opt.m/2):row_hat+floor(opt.m/2),col_hat-floor(opt.m/2):col_hat+floor(opt.m/2));
%     full_reconst_max(inds{1},inds{2}) = full_reconst_max(inds{1},inds{2}) + rl{opt_tmp.mom}(1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2),c1);
%   end
% 
%   figure(6); imagesc(full_reconst_max); axis image; colorbar;
% end

% Show ROIs extracted
% figure(7); imagesc(ROI_mask); axis image; colorbar;
% 
% 
% 
% figure(8); imagesc(orig_ROI_mask); axis image; colorbar;
% 
% figure(9); imagesc(model.y_orig); axis image; colorbar;
% figure(10); imagesc(model.y); axis image; colorbar;