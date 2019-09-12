function [ out ] = symmetrise( inp )
%SYMMETRISE Symmetrises an n-dimensional hypercube by rotating through it's
%dimensions, summing up the objects then dividing by the number of
%dimensions.


out = zeros(size(inp));

all_perms = perms(1:ndims(inp));

for j1 = 1:size(all_perms,1)
  out = out + permute(inp,all_perms(j1,:));
end

out = out./size(all_perms,1);




% % Old incorrect version (works up to order 3, incorrect in order 4!
% out = zeros(size(inp));
% for j1 = 0:(ndims(inp)-1)
%   out = out + shiftdim(inp,j1);
% end
% 
% out = out./ndims(inp);


end

