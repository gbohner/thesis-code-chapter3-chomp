
% Assumes we have everything loaded (from examine_results), and mom1
% supplied

figure(711); imagesc(opt.cumulant_pixelwise{mom1});

figure(712); im_handle = imagesc(opt.cumulant_pixelwise{mom1});
full_reconst = zeros(szWY(1:2));
for c1 = use_cells
  row_hat = H(c1,1); col_hat = H(c1,2);
  [inds, cut] = mat_boundary(size(full_reconst),row_hat-floor(opt.m/2):row_hat+floor(opt.m/2),col_hat-floor(opt.m/2):col_hat+floor(opt.m/2));
  full_reconst(inds{1},inds{2}) = full_reconst(inds{1},inds{2}) + rl{mom1}(1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2),c1);
  
  set(im_handle, 'CData', (opt.cumulant_pixelwise{mom1} - full_reconst));
  drawnow;
  pause(0.1);
end