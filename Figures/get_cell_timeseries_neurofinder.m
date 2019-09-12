function [timeseries, raw_mean, raw_var] = get_cell_timeseries_neurofinder(Yraw, ROIs, use_frames, neurofinder_ROI_format)
% Returns the timeseries from ROIs, but also computes an online estimate of
% mean and variance of the raw data stack (using Chan1979
% https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm)


%% Loading

%Load the input file we wanna use (done outside manual for now)
%Yraw = chomp_data(get_path(opt, 'raw_virtual_stack'));

%% Getting timeseries from ROIs
szRaw = size(Yraw);

if nargin < 3, use_frames=[]; end
if isempty(use_frames)
  use_frames = 1:szRaw(3);
end

if nargin < 4, neurofinder_ROI_format=[]; end
if isempty(neurofinder_ROI_format), neurofinder_ROI_format = 0; end


% Stack all ROIs
stackedROIs = zeros(length(ROIs), szRaw(1),szRaw(2)); % will be used to project ROIs into scalar per ROI

if ~neurofinder_ROI_format % CHOMP ROI format
  cutsize = size(ROIs{1}.mask,1);
  for i1 = 1:length(ROIs)
    [inds, cut] = mat_boundary(szRaw(1:2),ROIs{i1}.row-floor(cutsize/2):ROIs{i1}.row+floor(cutsize/2),ROIs{i1}.col-floor(cutsize/2):ROIs{i1}.col+floor(cutsize/2));
    cur_ROI_mask = ROIs{i1}.mask(1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2));
    stackedROIs(i1,inds{1},inds{2}) = 1.*cur_ROI_mask; 
  end
else
  orig_ROIs = ROIs;
  for i1 = 1:length(orig_ROIs)
    for j1 = 1:size(orig_ROIs{i1},1)
      stackedROIs(i1, orig_ROIs{i1}(j1,1), orig_ROIs{i1}(j1,2)) = 1.;
    end
  end
end

stackedROIs = reshape(stackedROIs,size(stackedROIs,1),[]);

% Normalise each ROI
stackedROIs = stackedROIs./sum(stackedROIs,2); % if all positive, this works, otherwise maybe divide by norm?

stackedROIs = sparse(stackedROIs); % faster multiplication

% Initialise online mean and var calculations
avg_a = 0;
count_a = 0;
var_a = 0;

timeseries = nan(size(stackedROIs,1), max(use_frames));
for t1 = use_frames
  cur_im = Yraw(:,:,t1); % load image
  timeseries(:,t1) = stackedROIs*cur_im(:);
  
  % Get current mean and var
  avg_b = mean(cur_im(:));
  var_b = var(cur_im(:), 0);% Matlab computes sample variance (n-1)
  count_b  = numel(cur_im(:));
  
  % Update mean and var and count
  if t1 ~= use_frames(1)
    delta = avg_b - avg_a;
    m_a = var_a * (count_a - 1);
    m_b = var_b * (count_b - 1);
    M2 = m_a + m_b + (delta.^2 * count_a * count_b) / (count_a + count_b);
    var_a = M2 / (count_a + count_b - 1);
    avg_a = (count_a * avg_a + count_b * avg_b)./(count_a+count_b);
    count_a = count_a + count_b;
  else
    count_a = count_b;
    avg_a = avg_b;
    var_a = var_b;
  end
end

raw_mean = avg_a;
raw_var = var_a;

end