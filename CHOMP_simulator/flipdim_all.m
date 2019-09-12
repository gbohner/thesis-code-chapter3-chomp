function [ T ] = flipdim_all( T, varargin )
%FILPDIM_ALL Flips all dimensions for an nth order tensor
%   Useful for use with convn (generalization of rot90(T,2)), for computing
%   correlation instead of convolution
%   With additional argument one can specify which dimensions to flip

if nargin <= 1
  dims_to_flip = 1:numel(size(T));
elseif nargin>1
  dims_to_flip = varargin{1};
end
  
for dim = dims_to_flip
  T = flipdim(T,dim);
end


end

