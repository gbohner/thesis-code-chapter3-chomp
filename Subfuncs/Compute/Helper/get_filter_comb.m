function filt = get_filter_comb(W, filt_comb, varargin)
  % Given a set of filters and a row of all_combs, return the requested combination
  if nargin>2
    using_diag_cumulants = varargin{1};
  else
    using_diag_cumulants = 0;
  end
  
  if ~using_diag_cumulants
    filt = W(:, filt_comb(1));
    for i11 = 2:size(filt_comb,2)
      % Make sure everything we do is (super)symmetric!
      %OLD VERSION HAS 0 in mply, might be wrong? filt = symmetrise(mply(filt,  W(:, filt_comb(i11))', 0)); % always add an extra dim by multiplying from the right with a row vector
      filt = symmetrise(mply(filt,  W(:, filt_comb(i11))', 1)); % always add an extra dim by multiplying from the right with a row vector
    end
  else % Do not create a high dimensional tensor, but rather multiply the filters pointwise
    mom_comb = zeros(1,size(W,2));
    mom_comb = histc(filt_comb(:)',1:size(mom_comb,2));
    filt = prod(bsxfun(@power, W, mom_comb), 2);
  end
end