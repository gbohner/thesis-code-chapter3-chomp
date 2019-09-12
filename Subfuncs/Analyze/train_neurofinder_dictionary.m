function [W, W_weights] = train_neurofinder_dictionary( opt, is_preproc2P )
%TRAIN_NEUROFINDER_ROIS Train a model that subselects and predicts shape
%for neurofinder ROIs

if nargin < 2
    is_preproc2P = false;
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


all_orig_H = size(orig_H,1);
% Reject cells that are too close to one another
closest_neigh = min(squareform(pdist(orig_H(:,1:2))) + diag(Inf*ones(size(orig_H,1),1)),[],2);
if ~strcmp(opt.learn_decomp, 'NMF')
  orig_H = orig_H((closest_neigh) > (opt.m*0.8),:);
else % for NMF we can keep cells closer by for training
  orig_H = orig_H((closest_neigh) > (opt.m*0.2),:);
end
kept_orig_H = size(orig_H,1);
fprintf('\nTraining supervised neurofinder dictionary, using %d/%d cells...\n', kept_orig_H, all_orig_H)

% Load the data
load(get_path(opt),'inp');

W_weights = inp.opt.W_weights;


for type = 1:inp.opt.NSS
  [W(:,inp.opt.Wblocks{type}), use_cost] = update_dict(inp.data,orig_H, zeros(inp.opt.m^2, length(opt.Wblocks{type})), inp.opt,inp.opt.KS+3,type);
  if strcmp(opt.W_weight_type, 'decomp')
    W_weights(inp.opt.Wblocks{type}) = use_cost;
  end
end

end

