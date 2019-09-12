function update_visualize_model(model, varargin)

  if nargin<2 %~exist('use_cells', 'var')
    use_cells = 1:size(model.H,1);
  else
    use_cells = varargin{1};
  end
  if nargin<3
    show_numbers = 0;
  else
    show_numbers = varargin{2};
  end
  
  update_visualize(model.y,model.H(use_cells,:), ...
  reshape(model.W,model.opt.m,model.opt.m,size(model.W,ndims(model.W))),...
  model.opt,1,show_numbers);
end
