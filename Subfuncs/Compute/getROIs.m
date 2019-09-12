function [ ROI_image, ROIs ] = getROIs( opt, varargin )
%GETROIS Summary of this function goes here
%   Detailed explanation goes here

load(get_path(opt, 'output_iter', opt.niter), 'model');
 [H, W, X, y_orig, y] = model.get_fields( 'H', 'W', 'X', 'y_orig','y');

%update_visualize( y_orig,H,reshape(W,opt.m,opt.m,size(W,2)),opt,1);

if nargin>1
  to_reconst = varargin{1};
  num_reconst = max(varargin{1});
else
  to_reconst = 1:size(H,1);
  num_reconst = size(H,1);
end

if nargin>2
  if varargin{2}
    %Get random ROIs
    H = floor(1+rand(size(H,1)).*numel(y));
    X = randn(size(X));
  end
end

sz = size(y);

ROIs = cell(num_reconst,1);
switch opt.ROI_type
    case 'quantile'
      ROI_image = zeros([sz(1:2), 3]);
    case 'quantile_dynamic'
      ROI_image = zeros([sz(1:2), 3]);
    case 'ones_origsize'
      szRaw = size(y_orig);
      ROI_image = zeros([szRaw(1:2), 3]);
    case 'quantile_origsize'
      szRaw = size(y_orig);
      ROI_image = zeros([szRaw(1:2), 3]);
    case 'quantile_dynamic_origsize'
      szRaw = size(y_orig);
      ROI_image = zeros([szRaw(1:2), 3]);
    case 'mean_origsize'
      szRaw = size(y_orig);
      ROI_image = zeros([szRaw(1:2), 3]);
    case 'brightest_pixel_origsize'
      szRaw = size(y_orig);
      ROI_image = zeros([szRaw(1:2), 3]);
end


Wfull = cell(opt.NSS,1);
for obj_type = 1:opt.NSS
  [~,~,Wfull_cur] = reconstruct_cell( opt, W(:,opt.Wblocks{obj_type}), X(1,:), 'do_subset_combs', false, 'do_reconstruction', false);
  Wfull{obj_type} = Wfull_cur;
end


% [all_reconst,all_reconst_lowdim] = reconstruct_cell( opt, W, X);

for i1 = to_reconst
  row = H(i1, 1); col = H(i1, 2); type = H(i1, 3);
%   if opt.mom>=2 %Then reconstruct variance image
%     reconst = all_reconst_lowdim{2}(:,:,i1);
%   else
  [reconst_full, reconst_lowdim, ~] = reconstruct_cell(...
        opt, W(:,opt.Wblocks{type}), X(i1,:),'Wfull',Wfull{type});


  % Use variance reconstruction if available
  reconst = reconst_lowdim{min(length(reconst_lowdim),2)}(:,:,1);
 
  reconst_orig = reconst;
  
  if opt.fig>=4
    figure; imagesc(reconst); pause(0.2);
  end
  
  switch opt.ROI_type
    case 'quantile'
      reconst = reconst > quantile(reconst(:),opt.ROI_params(1));
      reconst = bwconvhull(reconst);
      [inds, cut] = mat_boundary(sz(1:2),row-floor(opt.m/2):row+floor(opt.m/2),col-floor(opt.m/2):col+floor(opt.m/2));
    case 'quantile_dynamic'
      reconst = abs(reconst);
      tmp = opt.ROI_params(1)*(max(reconst(:))-min(reconst(:)))+min(reconst(:));
      tmp = sum(reconst(:)>tmp)./numel(reconst(:)); %ratio of pixels to pick
      [reconst_vals, tmp_ind] = sort(reconst(:),'descend');
      last_val = reconst_vals(floor(tmp*numel(reconst)));
      last_ind = find(reconst_vals == last_val,1,'last');
      reconst(:) = 0;
      reconst(tmp_ind(1:last_ind))=1;
      [inds, cut] = mat_boundary(sz(1:2),row-floor(opt.m/2):row+floor(opt.m/2),col-floor(opt.m/2):col+floor(opt.m/2));
    case 'ones_origsize'
      cutsize = round(opt.m./opt.spatial_scale);
      cutsize = cutsize + (1-mod(cutsize,2));
      reconst = ones([cutsize,cutsize]);
      row = round((row-1)./opt.spatial_scale)+1;
      col = round((col-1)./opt.spatial_scale)+1;
      [inds, cut] = mat_boundary(szRaw(1:2),row-floor(cutsize/2):row+floor(cutsize/2),col-floor(cutsize/2):col+floor(cutsize/2));
    case 'quantile_origsize'
      reconst = abs(reconst);
      cutsize = round(opt.m./opt.spatial_scale);
      cutsize = cutsize + (1-mod(cutsize,2));
      reconst = imresize(reconst,[cutsize,cutsize]);
      reconst = reconst > quantile(reconst(:),opt.ROI_params(1));
      reconst = imerode(reconst,strel('rectangle',[3,3]));
      reconst = imdilate(reconst,strel('rectangle',[3,3]));
      reconst = imdilate(reconst,strel('rectangle',[3,3]));
      reconst = imerode(reconst,strel('rectangle',[3,3]));
      reconst = imerode(reconst,strel('rectangle',[3,3]));
      %reconst = bwconvhull(reconst); %looks better for visualizations
      row = round((row-1)./opt.spatial_scale)+1;
      col = round((col-1)./opt.spatial_scale)+1;
      [inds, cut] = mat_boundary(szRaw(1:2),row-floor(cutsize/2):row+floor(cutsize/2),col-floor(cutsize/2):col+floor(cutsize/2));
    case 'mean_origsize'
      cutsize = round(opt.m./opt.spatial_scale);
      cutsize = cutsize + (1-mod(cutsize,2));
      reconst = imresize(reconst,[cutsize,cutsize]);
      %reconst = reconst > quantile(reconst(:),opt.ROI_params(1));
      %reconst = bwconvhull(reconst); %looks better for visualizations
      row = round((row-1)./opt.spatial_scale)+1;
      col = round((col-1)./opt.spatial_scale)+1;
      [inds, cut] = mat_boundary(szRaw(1:2),row-floor(cutsize/2):row+floor(cutsize/2),col-floor(cutsize/2):col+floor(cutsize/2));
%       %get the covariance between pixels of reconstruction and original
%       %mean
%       reconst = reconst(1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2)) .* y_orig(inds{1},inds{2});
    case 'quantile_dynamic_origsize'
      reconst = abs(reconst);
      cutsize = round(opt.m./opt.spatial_scale);
      cutsize = cutsize + (1-mod(cutsize,2));
      reconst = imresize(reconst,[cutsize,cutsize]);
      row = round((row-1)./opt.spatial_scale)+1;
      col = round((col-1)./opt.spatial_scale)+1;
      [inds, cut] = mat_boundary(szRaw(1:2),row-floor(cutsize/2):row+floor(cutsize/2),col-floor(cutsize/2):col+floor(cutsize/2));
      tmp = opt.ROI_params(1)*(max(reshape(y_orig(inds{1},inds{2}),1,[])')-min(reshape(y_orig(inds{1},inds{2}),1,[])'))+min(reshape(y_orig(inds{1},inds{2}),1,[])');
      tmp = sum(sum(y_orig(inds{1},inds{2})>tmp))./numel(y_orig(inds{1},inds{2})); %ratio of pixels to pick
      [reconst_vals, tmp_ind] = sort(reconst(:),'descend');
      last_val = reconst_vals(floor(tmp*numel(reconst)));
      last_ind = find(reconst_vals == last_val,1,'last');
      reconst(:) = 0;
      reconst(tmp_ind(1:last_ind))=1;
    case 'brightest_pixel_origsize'
      reconst = 1.*(reconst==max(reconst(:)));
      cutsize = round(opt.m./opt.spatial_scale);
      cutsize = cutsize + (1-mod(cutsize,2));
      reconst = imresize(reconst,[cutsize,cutsize]);
      row = round((row-1)./opt.spatial_scale)+1;
      col = round((col-1)./opt.spatial_scale)+1;
      [inds, cut] = mat_boundary(szRaw(1:2),row-floor(cutsize/2):row+floor(cutsize/2),col-floor(cutsize/2):col+floor(cutsize/2));
    otherwise
      error('CHOMP:roi:method',  'Region of interest option string (opt.ROI_type) does not correspond to implemented options.')
  end
  
  
  if opt.W_force_round
    [~, mask1] = transform_inds_circ(0,0,150,size(reconst,1),(size(reconst,1)-1)/2,0);
    reconst = reconst.*mask1;
    [~, mask2] = transform_inds_circ(0,0,150,size(reconst_orig,1),(size(reconst_orig,1)-1)/2,0);
    reconst_orig = reconst_orig.*mask2;
  end
  
  % Store all ROIs as a color image, with each ROI having a random color
  ROI_image(inds{1},inds{2}, :) = ROI_image(inds{1},inds{2}, :) + ...
    mply(reconst(1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2)), 0.7*randn(1,3));
  
  %Store results in cells now, later add option for json output %TODO
  ROIs{i1} = struct('col', col, 'row', row, 'type', type, 'mask', reconst, 'reconst', reconst_orig);
  
  
  
  
end

  if opt.fig >1
    figure(2);
    imshow(ROI_image); axis square;
  end

end

