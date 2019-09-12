function out = get_patch( stack, opt, H, varargin )
%GET_PATCH Gets patch(es) from the data at given location at time
%   get_patch(data,opt,H) - returns an array of patch time blocks
%   around locations H, size specified as in opt.m, out = row x col x t x
%   cell_num
%   get_patch(data,opt,H, t) - returns a cell array of patches blocks
%   around locations H at time t, size specified as in opt.m

szRaw = chomp_size(stack,'Y');

p = inputParser();
p.addRequired('stack',@isstruct);
p.addRequired('opt',@(x)isa(x,'chomp_options'));
p.addRequired('H',@isnumeric);
p.addOptional('times',1:szRaw(end),@isnumeric);
p.addParameter('scaled',0,@isnumeric);
p.addParameter('minDiskAccess',0,@isnumeric);
p.parse(stack,opt,H,varargin{:});


if ~p.Results.scaled
  szPatch = [opt.m, opt.m];
  szY = szRaw;
else
  tmp = floor(opt.m./opt.spatial_scale)+1-mod(floor(opt.m./opt.spatial_scale),2);
  szPatch = [tmp tmp];
  szY = round(szRaw*opt.spatial_scale);
end

%Initialize output array
out = cell(numel(p.Results.times),1);
[out{:}]= deal(zeros([numel(H),szPatch]));


for t = p.Results.times %over frames required
  frame_t = stack.Y(:,:,t);
  for i1 = 1:numel(H)
    [row, col, type] = ind2sub([szY(1:2) opt.NSS],H(i1));
    if p.Results.scaled, row = round(row./opt.spatial_scale); col = round(col./opt.spatial_scale); end
    [ valid_inds, cuts ] = mat_boundary(szRaw(1:2), row-floor(szPatch(1)/2):row+floor(szPatch(1)/2), col-floor(szPatch(1)/2):col+floor(szPatch(1)/2));
    out{t}(i1,1+cuts(1,1):end-cuts(1,2),1+cuts(2,1):end-cuts(2,2)) = frame_t(valid_inds{1},valid_inds{2});
  end
end

out = reshape(cell2mat(out),[numel(H),numel(p.Results.times),szPatch]);
out = permute(out, [3,4,2,1]); % Change to row x col x t x cell_num
end

