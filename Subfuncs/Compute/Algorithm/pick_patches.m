function [out, num_cells, col_count] = pick_patches( datas, Hs, opts,type, varargin)
%PICK_PATCHES Returns patches to learn from, Y is the data tensor, H are the spatial locations, type is cell type number

%Make sure all the input are cell arrays (of possibly multiple datasets)
if ~iscell(datas), datas = {datas}; end
if ~iscell(Hs), Hs = {Hs}; end
if ~iscell(opts), opts = {opts}; end

do_cov = 2; 
% 0 - return all n order patches, 
% 1 - return covariance matrix build from n-order patches, 
% 2 - return uncentered covariance matrix built from raw patches
% 3 - return raw patches (width x height x t x cells)

if nargin > 4 %Check if we want to just output the covariance matrix instantly
  do_cov = varargin{1};
end

py = cell(numel(datas),1);
num_cells = zeros(numel(datas,1));

for c1 = 1:numel(datas)
  data = datas{c1};
  H = Hs{c1};
  opt = opts{c1};
  
  szY = chomp_size(data.proc_stack,'Y');
  
  %Remove entries from H that are the wrong type
  h1 = 1;
  while h1<=size(H,1)
    cur_type = H(h1,3);
    if cur_type ~= type, H(h1,:) = []; else h1 = h1+1; end
  end
  
  
  if do_cov
    py{c1} = struct('mat',zeros(opt.m^2), 'count', 0);
  else
    py{c1} = cell(size(H,1),1);
  end
  
  if ~isempty(H)    
    % Re-center ROIs (modify H so that is in the centre of mass of
    % the reconstructed binary ROI)
    tmp_opt = struct_merge( chomp_options(), opt ); % Essentially creates a deepcopy
    tmp_opt.ROI_type = 'quantile_dynamic';
    tmp_opt.ROI_params = 0.7;
    % Find latest completed iteration
    completed_iter_names = dir([tmp_opt.output_folder tmp_opt.file_prefix '_' tmp_opt.timestamp '*']);
    if ~isempty(completed_iter_names)
      tmp_opt.niter = str2num(completed_iter_names(end).name(strfind(completed_iter_names(end).name, '_iter_')+6:end-4)); 
      [ ROI_image, ROIs ] = getROIs( tmp_opt );
      for i11 = 1:size(H,3)
        [rows, cols] = find(ROIs{i11}.mask);
        rows = rows + ROIs{i11}.row - (size(ROIs{i11}.mask,1)+1)/2;
        cols = cols + ROIs{i11}.col - (size(ROIs{i11}.mask,2)+1)/2;
        H(i11,1:2) = round(mean([rows(:), cols(:)],1)); % Re-centering ROIs
      end
    end
    % Reject cells too close to the borders
    H(((H(:,1)<opt.m) + (H(:,1)>(szY(1)-opt.m))+(H(:,2)<opt.m)+(H(:,2)>(szY(2)-opt.m)))>0,:) = [];
    
    
    patches = get_patch(data.proc_stack, opt, sub2ind([szY(1:2) opt.NSS], H(:,1), H(:,2), H(:,3)));
    
    if opt.W_addflat
      patches = bsxfun(@minus, patches, mean(mean(patches,2),1));
    end
    
    if opt.W_force_round
      [~, mask] = transform_inds_circ(0,0,150,opt.m,(opt.m-1)/2,0);
      patches = patches.*mask;
    end
    
    
  else
    continue;
  end
  
  num_cells(c1,1) = size(H,1);
  
  
  if do_cov == 2 %Just build the covariance matrix out of the raw patch samples
    patches = reshape(patches,opt.m^2,[]);
    %patches = patches - get_hist_mode(opt.A(:), 500); % Remove the mode of the mean
    %patches = bsxfun(@minus, patches, mean(patches,2)); % Commented out
    %20190515, as I believe we want to learn the mean signal mode too!
    py{c1}.count = size(patches,2);
    py{c1}.mat = (patches * patches');
  elseif do_cov == 3 % Return raw patches (space x time x cell_id)
    py{c1} = reshape(patches,[opt.m^2,size(patches,3),size(patches,4)]);
  else
    %Process the individual patches to get n-order estimates
    for h1 = 1:size(H,1)
      %disp(h1);
      curpy = get_n_order_patch(patches(:,:,:,h1), opt, szY);
      if do_cov
        %Also weigth every moment tensor according to the number of independent
        %elements (patchsize multichoose mom) over total number of elements
        for mom1 = 1:opt.mom
          curpy{mom1} = curpy{mom1} .* (nchoosek(opt.m^2+mom1-1,mom1)./((opt.m^2).^mom1));
        end
        
        %Just store the resulting covariance matrix
        for mom1 = 1:opt.mom
          py{c1}.mat = py{c1}.mat + ...
            reshape(curpy{mom1},size(curpy{mom1},1),[])*reshape(curpy{mom1},size(curpy{mom1},1),[])'; %reshape to flat for "HOSVD"
          py{c1}.count = py{c1}.count + size(reshape(curpy{mom1},size(curpy{mom1},1),[]),2); %number of columns
        end
      else
        %Store all the individual, weighted and flattened vectors
        py{c1}{h1} = curpy;
      end
    end
  end


end

opt = opts{1};

if do_cov == 1 || do_cov == 2
  out = zeros(opt.m^2, opt.m^2);
  col_count = 0;
  for c1 = 1:numel(py)
    %Combine all the info from all datasets, with numerical stability
    weigth = (py{c1}.count - 1);
    out = out + py{c1}.mat./weigth; %number of samples
    col_count = col_count + py{c1}.count;
  end
elseif do_cov == 3
  out = [];
  for c1 = 1:numel(datas)
    out = cat(3, out, py{c1});
  end
  num_cells = size(out,3);
  col_count = size(out,2)*size(out,3);
  return
else
  %Concatanate the results into a
  out={};
  for c1 = 1:numel(py)
    out(end+1:end+numel(py{c1})) = py{c1};
  end

  out = flatten_patches(out, opt); % opt.m^2 x (location*(opt.m^2)^opt.mom)  - very flat matrix
  col_count = size(out,2);
end

num_cells = sum(num_cells);


function py=flatten_patches(patches,opt)
  py = [];
  for i1 = 1:length(patches)
      out1 = [];
      patch = patches{i1};
      for mom = 1:opt.mom
        cur = patch{mom};
        cur = reshape(cur,opt.m^2,[]);
        cur = cur./size(cur,2); %normalize by dimensionality
        out1 = [out1, cur];
      end
      py(:,end+1:end+size(out1,2)) = out1;
  end
end

end