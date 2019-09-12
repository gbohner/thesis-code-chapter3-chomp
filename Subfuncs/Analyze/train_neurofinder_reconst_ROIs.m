function trainedModel = train_neurofinder_reconst_ROIs( opt, is_preproc2P, train_offdiag )
%TRAIN_NEUROFINDER_ROIS Train a model that subselects and predicts shape
%for neurofinder ROIs

if nargin < 2
    is_preproc2P = false;
end

if nargin <3
  train_offdiag = false;
end

if is_preproc2P
    orig_path = fileparts(fileparts(fileparts(fileparts(get_path(opt)))));
else
    orig_path = fileparts(fileparts(fileparts(get_path(opt))));
end

% Load the training data
%orig_regions = py.neurofinder.load([orig_path filesep 'regions' filesep 'regions.json']);
%orig_ROIs = cellfun(@matpy.nparray2mat, cell(orig_regions.coordinates), 'UniformOutput', false);
orig_regions = loadjson([orig_path filesep 'regions' filesep 'regions.json']);
orig_ROIs = cellfun(@(x)x.coordinates+1, orig_regions, 'UniformOutput', false);


% Get the centers for the training ROIs
orig_ROI_centers = cellfun(@(X)round(mean(X*opt.spatial_scale))', orig_ROIs, 'UniformOutput', false);
orig_H = [cell2mat(orig_ROI_centers)' ones(length(orig_ROI_centers),1)];

% Load our predicted ROIs
use_iter = opt.niter;

% Load model for use_iter
load(get_path(opt,'output_iter',use_iter),'model');

[H, W, X, y_orig, y, L] = model.get_fields( 'H', 'W', 'X', 'y_orig','y','L');
szWY = size(model.y);

% % Load the data
% load(get_path(opt),'inp');
% data = inp.data;

% if ~exist('use_cells', 'var')
%   use_cells = 1:size(H,1);
% end
      
% [ROI_mask, ROIs] = getROIs(model.opt);


rand_H = min(H(:,1:2),[],1) + floor((max(H(:,1:2),[],1) - min(H(:,1:2),[],1)).*rand(size(H,1),2));

all_H = [orig_H(:,1:2); H(:,1:2); rand_H(:,1:2)];

% Get the pairwise ROI distance matrix between true and current ROIs, as
% well as some random locations
[knn_inds, knn_dists] = knnsearch(orig_H(:,1:2), all_H, 'K', 1);

% Create a classifier from X to distance to nearest actual ROI (so
% basically whether we can learn from reconstructuion weigths whether its a
% true ROI or not)
% classifTargets = (exp(-knn_dists/(opt.m/5)))>0.5

%classifTargets = knn_dists <= 5*opt.spatial_scale;
classifTargets = exp(-knn_dists ./ (opt.m./3.));
classifTargetsBin = classifTargets>=(exp(-0.8));

if ~train_offdiag
  full_feature_im = cat(3, opt.cumulant_pixelwise{(1+opt.diag_cumulants_offdiagonly):opt.mom}); % If opt.diag_cumulants_offdiagonly, skip the first moment, as it has no offdiagonal
  
  % Subtract the mode (not reconstructed for now!) from each feature_im
  mom_mode = [];
  for mom1 = 1:opt.mom, mom_mode(mom1) = get_hist_mode(opt.cumulant_pixelwise{mom1},500); end;
  full_feature_im = full_feature_im - reshape(mom_mode((1+opt.diag_cumulants_offdiagonly):opt.mom),[1,1,size(full_feature_im,3)]);
  
  % Get the cumulant images as classifFeatures
  classifFeatures = zeros(size(all_H,1), opt.m^2 * size(full_feature_im,3));
  
  for i1 = 1:size(all_H,1)
    featureIm = nan(opt.m, opt.m, size(full_feature_im,3));

    % Get feature coordinates
    row = all_H(i1, 1); col = all_H(i1, 2);
    [inds, cut] = mat_boundary(szWY(1:2),row-floor(opt.m/2):row+floor(opt.m/2),col-floor(opt.m/2):col+floor(opt.m/2));
    featureIm(1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2), :) = full_feature_im(inds{1},inds{2}, :);

    classifFeatures(i1, :) = featureIm(:);
  end
else % Take the full tensors at every location
  num_features = sum((opt.m^2).^((1+opt.diag_cumulants_offdiagonly):opt.mom));
  classifFeatures = zeros(size(all_H,1), num_features);
  inp = load(get_path(opt));
  data = inp.inp.data;
  szY = chomp_size(data.proc_stack,'Y');
  patches = get_patch(data.proc_stack, opt, sub2ind([szY(1:2)], all_H(:,1), all_H(:,2)));

  for h1 = 1:size(H,1)
      %disp(h1);
      cur_cumulants = get_n_order_patch(patches(:,:,:,h1), opt, szY); % Cumulant tensors
      cur_ind = 0;
      for mom1 = (1+opt.diag_cumulants_offdiagonly):opt.mom
        tmp_cum = cur_cumulants{mom1}(:); % TODO - remove the diagonals from these later if opt.diag_cumulants_offdiagonly, for precision
        classifFeatures(h1,(cur_ind+1):(cur_ind+numel(tmp_cum))) = tmp_cum;
        cur_ind = cur_ind + numel(tmp_cum);
      end
  end
    
end


classifModel = TreeBagger(30, classifFeatures, classifTargets, 'Method', 'regression');
trainedModel{1} = classifModel;


%% Given the selected ROIs, learn how to map the raw data around suggested center to the binary ROI


% Get the reconstruction around matched locations
regrTargetSizeOrig = int16(((opt.m/opt.spatial_scale)) + 1 - mod((opt.m/opt.spatial_scale),2));
regrTargetSize = 7;
regrTargets = zeros(size(orig_H,1), regrTargetSize^2);
regrModels = cell(regrTargetSize,1);


regrFeatures = classifFeatures(1:size(orig_H,1), :);


for i1 = 1:size(orig_H,1)
  % Get feature coordinates
  row = orig_H(i1, 1); col = orig_H(i1, 2);
  
  % Get target coordinates
  row_origsize = round((row-1)./opt.spatial_scale)+1;
  col_origsize = round((col-1)./opt.spatial_scale)+1;
  targetCoords = orig_ROIs{i1} + (double(regrTargetSizeOrig)+1)/2 - repmat([row_origsize col_origsize], size(orig_ROIs{i1},1),1);
  j1 = 1;
  while j1 <= size(targetCoords, 1)
    if (targetCoords(j1,1) < 1) || (targetCoords(j1,2) < 1) || (targetCoords(j1,1) > regrTargetSizeOrig) || (targetCoords(j1,2) > regrTargetSizeOrig)
      targetCoords(j1,:) = [];
    else
      j1 = j1+1;
    end
  end
  targetIm(sub2ind([regrTargetSizeOrig, regrTargetSizeOrig], targetCoords(:,1), targetCoords(:,2))) = 1;

  targetImSmall = imresize(targetIm, [regrTargetSize, regrTargetSize], 'nearest');
  
  regrTargets(i1,:) = targetImSmall(:);
  
%   if opt.fig > 3
%     figure(117); 
%     subplot(1,2,1); imagesc(featureIm); axis image;
%     subplot(1,2,2); imagesc(targetIm); axis image;
%     pause;
%   end
end
  


parfor m1 = 1:(regrTargetSize^2)
  %regrModels{m1} = TreeBagger(opt.m^2, regrFeatures, regrTargets(:,m1));
  regrModels{m1} = TreeBagger(10, regrFeatures, regrTargets(:, m1), 'Method', 'regression');
end

trainedModel{2} = regrModels;

end

