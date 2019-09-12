%dataset_name = 'neurofinder.01.00'; use_cells = 1:250; % testing stuff
%chomp_iter = '4';

load(get_path(opt));
curY = inp.data.proc_stack.Y;

x = [];

% pixels_to_use = {[80,229],[81,229],[74,242], [92,229], [92,242]};
% 
% for i1 = 1:length(pixels_to_use)
%   x = [x, squeeze(curY(pixels_to_use{i1}(1),pixels_to_use{i1}(2),:))];
% end
% x = x'; % time is dim2

%x = squeeze(curY(72:99,232,:));
%x = squeeze(curY(232,[72:73,80:81,93:94],:));
%x = squeeze(curY(232,72:120,:));


xmoms = {};
for t = 1:size(x,2)
  curX = x(:,t);
  for mom1 = 1:4
    if t == 1, xmoms{mom1} = zeros([size(x,1)*ones(1,mom1),1]); end
    xmoms{mom1} = xmoms{mom1} + curX;
    curX = mply(curX,x(:,t)',1);
  end
end

for mom1 = 1:4
  xmoms{mom1} = xmoms{mom1}./size(x,2);
end

xcums = raw2cum_multivariate(xmoms);
    

figure; imagesc(xcums{2}); colorbar; axis image
figure; imagesc(reshape(xcums{3},[],size(x,1))); colorbar; axis image
figure; imagesc(reshape(xcums{4},[],size(x,1)*size(x,1))); colorbar; axis image
%figure; imagesc(reshape(xcums{4}./symmetrise(mply(xcums{2},shiftdim(xcums{2},-1),1)),[],size(x,1)*size(x,1)),[-10,10]); colorbar

%save('Examples/data01_xcum_examples_20190605', 'xcums','pixels_to_use','opt')