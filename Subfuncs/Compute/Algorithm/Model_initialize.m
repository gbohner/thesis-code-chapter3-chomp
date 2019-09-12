function [W,  W_weights]  = Model_initialize( opt )
%MODEL_INITIALIZE Initializes the basis function matrices
%   Detailed explanation goes here
  
  W = zeros(opt.m^2, opt.NSS*opt.KS); % initialize basis functions
    % Initialize circles with reasonable size
  W_weights = opt.W_weights;
    
  for type = 1:opt.NSS
    init_model = opt.init_model{type};
    switch init_model
      case 'filled'
        [~, mask] = transform_inds_circ(0,0,150,opt.m,(opt.m-1)/2,0); % . , . , ., filter size, circle outer radius, inner hole radius
        W(:,opt.Wblocks{type}(1)) = mask(:);
      case 'neurofinder_preproc2P'
        [W, W_weights] = train_neurofinder_dictionary(opt, 1);
      
      case 'supervised'
        %learn the best basis functions from varargin{2}, which should be a
        %collection of examples or alternative a set of locations within
        %the dataset (in which case just call an update_dict with the
        %preset H)
        
      case 'pointlike'
        % Initialize the to a dot/small circle
        [~, mask] = transform_inds_circ(0,0,150,opt.m,1,0);
        W(:,opt.Wblocks{type}(1)) = mask(:);
      case 'donut'
        [~, mask] = transform_inds_circ(0,0,150,opt.m,(opt.m-1)/2,max((opt.m-7),2)/2); % . , . , ., filter size, circle outer radius, inner hole radius
        W(:,opt.Wblocks{type}(1)) = mask(:);
      case 'donut_two'
        [~, mask] = transform_inds_circ(0,0,150,opt.m,(opt.m-1)/2,max((opt.m-5),2)/2); % . , . , ., filter size, circle outer radius, inner hole radius
        W(:,opt.Wblocks{type}(1)) = mask(:);
        [~, mask] = transform_inds_circ(0,0,150,opt.m,(opt.m-5)/2,0); % . , . , ., filter size, circle outer radius, inner hole radius
        W(:,opt.Wblocks{type}(2)) = mask(:);
      case 'donut_four'
        [~, mask] = transform_inds_circ(0,0,150,opt.m,(opt.m-1)/2,max((opt.m-3),2)/2); % . , . , ., filter size, circle outer radius, inner hole radius
        W(:,opt.Wblocks{type}(1)) = mask(:);
        [~, mask] = transform_inds_circ(0,0,150,opt.m,(opt.m-3)/2,max((opt.m-7),2)/2); % . , . , ., filter size, circle outer radius, inner hole radius
        W(:,opt.Wblocks{type}(2)) = mask(:);
        [~, mask] = transform_inds_circ(0,0,150,opt.m,(opt.m-7)/2,max((opt.m-9),2)/2); % . , . , ., filter size, circle outer radius, inner hole radius
        W(:,opt.Wblocks{type}(3)) = mask(:);
        [~, mask] = transform_inds_circ(0,0,150,opt.m,(opt.m-9)/2,0); % . , . , ., filter size, circle outer radius, inner hole radius
        W(:,opt.Wblocks{type}(4)) = mask(:);
      case 'donut_conv'
        [~, mask_outer] = transform_inds_circ(0,0,150,opt.m,ceil((opt.m-opt.m/3)/2),max(ceil((opt.m-2*opt.m/3)/2),1)); % . , . , ., filter size, circle outer radius, inner hole radius
        [~, mask_inner] = transform_inds_circ(0,0,150,opt.m,ceil((opt.m-2*opt.m/3)/2),0); % . , . , ., filter size, circle outer radius, inner hole radius
        mask = mask_outer-0.5*mask_inner;
        mask(mask==0) = -0.1;
        mask = conv2(mask, fspecial('gaussian', opt.m, opt.smooth_filter_mean), 'same');
        W(:,opt.Wblocks{type}(1)) = mask(:);
      case 'donut_conv_new'
        [~, mask_outer] = transform_inds_circ(0,0,150,opt.m,ceil((opt.m-opt.m/3)/2),max(ceil((opt.m-2*opt.m/3)/2),1)); % . , . , ., filter size, circle outer radius, inner hole radius
        [~, mask_inner] = transform_inds_circ(0,0,150,opt.m,ceil((opt.m-2*opt.m/3)/2),0); % . , . , ., filter size, circle outer radius, inner hole radius
        mask = mask_outer-0.5*mask_inner;
        W(:,opt.Wblocks{type}(1)) = mask(:);
      case 'given'
        W(:,opt.Wblocks{type}) = opt.init_W(:,opt.Wblocks{type});
      otherwise
        error('CHOMP:learning:initialize', 'Model initialization option string (opt.init_model) does not correspond to implemented options.');
    end
  end
    
%      [~, mask] = transform_inds_circ(0,0,150,opt.m,(opt.m-3)/2,0); % . , . , ., filter size, circle outer radius, inner hole radius
%     W(:,2) = mask(:);
%      [~, mask] = transform_inds_circ(0,0,150,opt.m,(opt.m-5)/2,0); % . , . , ., filter size, circle outer radius, inner hole radius
%     W(:,3) = mask(:);
  
  
  Worig = W;
  
  %Project the assumed initial basis function into feature space
%   if opt.rand_proj
%     W = opt.P*W;
%   end
  
if opt.W_addflat && ~all(W(:,1)==W(1,1)) % in latter case it is already added by training
  W = [ones(opt.m^2,1),W(:,1:end-1)];
end

if opt.W_force_round
  [~, mask] = transform_inds_circ(0,0,150,opt.m,(opt.m-1)/2,0);
  W = W.*mask(:);
end

% Make sure that basis function column norms are ~1. 
for i1 = 1:size(W,2)
  W(:,i1) = W(:,i1)./(norm(W(:,i1))+1e-6);
end
  
 
  
end

