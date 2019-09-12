function [orig_ROIs, orig_H, orig_ROI_mask] = get_neurofinder_orig_ROIs(opt)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

orig_path = fileparts(fileparts(fileparts(fileparts(get_path(opt))))); % This only works for preproc2p, otherwise delete one fileparts
orig_regions = loadjson([orig_path filesep 'regions' filesep 'regions.json']);
orig_info = loadjson([orig_path filesep 'info.json']);
orig_ROIs = cellfun(@(x)x.coordinates+1, orig_regions, 'UniformOutput', false);

% Get the centers for the training ROIs
orig_ROI_centers = cellfun(@(X)round(mean((X-1)*opt.spatial_scale+1))', orig_ROIs, 'UniformOutput', false);
orig_H = [cell2mat(orig_ROI_centers)' ones(length(orig_ROI_centers),1)];
orig_ROI_mask = zeros([orig_info.dimensions(1), orig_info.dimensions(2), 3]);
for i1 = 1:size(orig_ROIs,2)
  cur_color = 0.5*rand(1,3);
  try
    for j1 = 1:size(orig_ROIs{i1},1)
      orig_ROI_mask(orig_ROIs{i1}(j1,1), orig_ROIs{i1}(j1,2),:) = cur_color;
    end
  end
end


end

