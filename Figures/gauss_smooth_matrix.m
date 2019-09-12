function out = gauss_smooth_matrix(inp, filt_sig, remove_diag, set_nandiag)
%GAUSS_SMOOTH_MATRIX Filters an input matrix with a Gaussian bump width filt_sig (if >0)
% Can also remove diagonals before filtering, and add them back as nan
% afterwards. If remove_diag > 1, removes nearby offdiagonals as well.

if nargin<3
  remove_diag = 0;
end

if numel(remove_diag)>1
  % Remove specific diagonals instead of A to B
  remove_these_diags = remove_diag;
  remove_diag = 2;
else
  remove_these_diags = 2:remove_diag;
end

if nargin<4
  set_nandiag = 1;
end


out = inp;
im_numel = ones(size(inp));

if filt_sig > 0
  lx = ceil(3*filt_sig);
  dt = (-lx:lx)';
  gauss_filt = exp(-dt.^2/(2*filt_sig^2)) * exp(-dt'.^2/(2*filt_sig.^2));
else
  gauss_filt = 1;
end


if remove_diag
  out = out - diag(diag(out));
  im_numel = im_numel-diag(diag(im_numel));
  if remove_diag > 1
    % Check if square
    if size(out,1)~=size(out,2), error('Only implemented for square matrices'); end
    m = size(out,1);
    % Remove lower diagonals
    for i1 = remove_these_diags
      out(i1:m+1:(end-((i1-1)*m))) = 0;
      im_numel(i1:m+1:(end-((i1-1)*m))) = 0;
    end
    % Remove upper diagonals
    for i1 = remove_these_diags
      out(((i1-1)*m+1):m+1:end) = 0;
      im_numel(((i1-1)*m+1):m+1:end) = 0;
    end
  end
end

im_numel = filter2(gauss_filt, im_numel, 'same');
out = filter2(gauss_filt, out, 'same');

out = out./im_numel;

if remove_diag
  % Put back main diagonal
  cur_diag = diag(inp);
  if set_nandiag, cur_diag = nan*cur_diag; end
  out = out - diag(diag(out)) + diag(cur_diag);
    
  % Put back other diagonals
  if remove_diag > 1
    % Check if square
    if size(out,1)~=size(out,2), error('Only implemented for square matrices'); end
    m = size(out,1);
    % Remove lower diagonals
    for i1 = remove_these_diags
      cur_diag = diag(inp, -(i1-1));
      if set_nandiag, cur_diag = nan*cur_diag; end
      out(i1:m+1:(end-((i1-1)*m))) = cur_diag;
    end
    % Remove upper diagonals
    for i1 = remove_these_diags
      cur_diag = diag(inp, (i1-1));
      if set_nandiag, cur_diag = nan*cur_diag; end
      out(((i1-1)*m+1):m+1:end) = cur_diag;
    end
  end
end
  
end

