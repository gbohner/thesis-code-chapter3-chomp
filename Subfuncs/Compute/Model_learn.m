function Model_learn( opt)
%MODEL_LEARN Summary of this function goes here
%   Detailed explanation goes here



load(get_path(opt), 'inp'); %could load opt as well, but it may have changed
inp.opt = struct_merge(inp.opt, opt);

if inp.opt.init_iter %If not 0 we aren't starting from scratch
  load(get_path(inp.opt,'output_iter',inp.opt.init_iter) ,'model')
  inp.opt.W_weights = model.opt.W_weights;
  if (inp.opt.init_iter+1)< inp.opt.niter && inp.opt.learn
      %Update the dictionary (the W filters)
      utic = tic;
      if inp.opt.verbose
        fprintf('Iteration %d/%d, updating dictionary...', inp.opt.init_iter, inp.opt.niter);
      end
      for type = 1:inp.opt.NSS
        [W(:,inp.opt.Wblocks{type}), Sv] = update_dict(inp.data,model.H,model.W,inp.opt,inp.opt.init_iter+2, type);
        if strcmp(opt.W_weight_type, 'decomp')
          inp.opt.W_weights(inp.opt.Wblocks{type}) = Sv;
        end
      end
      if inp.opt.verbose
        fprintf(' took %.2f seconds\n',toc(utic))
      end
  elseif ~isempty(inp.opt.init_W)
    W = inp.opt.init_W;
  else
    W = model.W;
  end
else %Starting from scratch
  [W, W_weights]  = Model_initialize(inp.opt);
  inp.opt.W_weights = W_weights;
end


ltic = tic;

%% Run the learning
for n = (inp.opt.init_iter+1):inp.opt.niter 
    
    itic = tic;
    
    if inp.opt.verbose
      fprintf('Iteration %d/%d, inferring cell locations...', n, inp.opt.niter);
    end
    %Compute convolution of Y with the filters as well as the "local Gram
    %matrices of filters to use in the matching pursuit step afterwards
    [WY, GW, WnormInv] = compute_filters(inp.data, W, inp.opt );
    
    %Infer the hidden variables (H - location, X - reconstruction weight, L - log likelihood gain)
    if ~inp.opt.mask
      [ H, X, L] = extract_coefs( WY, GW, WnormInv, W, inp.opt);
    else
      [ H, X, L] = extract_coefs( WY, GW, WnormInv, W, inp.opt, inp.UserMask);
    end

    %Save model from current iteration
    model = chomp_model(inp.opt,W,H,X,L,inp.y,inp.y_orig,inp.V);
    if ~isempty(inp.UserMask), model.UserMask = inp.UserMask; end
    save(get_path(inp.opt,'output_iter',n) ,'model')
    
    if inp.opt.verbose
      fprintf(' took %.2f seconds\n',toc(itic));
    end
    
    %Visualize model
    if inp.opt.fig >0
      update_visualize(model.y,model.H, ...
        reshape(model.W,model.opt.m,model.opt.m,size(model.W,ndims(model.W))),...
        model.opt,1);
    end

    if n < inp.opt.niter && inp.opt.learn
      utic = tic;
      %Update the dictionary (the W filters)
      if inp.opt.verbose
        fprintf('Iteration %d/%d, updating dictionary...', n, inp.opt.niter);
      end
      for type = 1:inp.opt.NSS
        [W(:,inp.opt.Wblocks{type}), Sv] = update_dict(inp.data,model.H,model.W(:,inp.opt.Wblocks{type}),inp.opt,2*n+2,type);
        if strcmp(opt.W_weight_type, 'decomp')
          inp.opt.W_weights(inp.opt.Wblocks{type}) = Sv;
        end
      end
      if inp.opt.verbose
        fprintf(' took %.2f seconds\n',toc(utic))
      end
    end
    
    %Visualize model
    if inp.opt.fig >0
      update_visualize(model.y,model.H, ...
        reshape(W,model.opt.m,model.opt.m,size(model.W,ndims(model.W))),...
        model.opt,1);
    end
     
    if inp.opt.verbose
      fprintf('Iteration %d/%d finished, elapsed time is %0.2f seconds\n', n, inp.opt.niter, toc(ltic))
    end

end

end

