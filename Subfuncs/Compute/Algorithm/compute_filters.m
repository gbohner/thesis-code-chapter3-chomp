function [WY, GW, WnormInv] = compute_filters(data, W, opt )
%COMPUTE_FILTERS Computes the correlation between filter and data, plus the
% original MAP coefficients
%   Detailed explanation goes here

%Load the shift tensor GPT (we dont really want to keep it memory all the time
if ~exist(get_path(opt, 'precomputed'),'file')
  GPT = precompute_shift_tensors(opt);
else
  load(get_path(opt, 'precomputed'), 'GPT');
end

szY = chomp_size(data.proc_stack,'Y'); %size of the data tensor

%% Initialisation of outputs

WY = cell(opt.NSS,1); % Projection of data onto basis functions
WnormInv = cell(opt.NSS,1);  % Inverse Interaction between basis functions
GW = cell(opt.mom,1); % Effect of a reconstruction onto nearby projections


for obj_type = 1:opt.NSS
  WY{obj_type} = cell(opt.mom,1);
  WnormInv{obj_type} = cell(opt.mom,1);
  for mom1 = 1:opt.mom
    if opt.diag_tensors 
      filter_combs = opt.KS; % Fast version where we don't use combinations of filter responses for estimation
    else
      filter_combs = nchoosek(opt.KS+mom1-1,mom1); % For each object type consider all combinations of filters; exploiting (super)symmetricity to reduce storage and computation
    end
    % dont do this here so raw2cum works, do it later
%     if opt.diag_cumulants_extradiagonal && mom1>=2 
%       filter_combs = filter_combs+opt.KS; % Add a seperate set of diagonal filters that only explain diagonal
%     end
    WY{obj_type}{mom1} = zeros([szY(1:2),filter_combs]); %For every location store the regression wrt all filter_combs (elements of the N_filter^mom1 projection tensor for the mom1-th moment
    WnormInv{obj_type}{mom1} = zeros([filter_combs, filter_combs]);
    GW{mom1} = cell([filter_combs*opt.NSS, filter_combs*opt.NSS]); % This may be HUGE too big for large moments and large filter sizes - %TODO store on disk
  end
end


%% Computing WY - projections

if ~opt.diag_cumulants || opt.diag_cumulants_offdiagonly % Compute cumulant tensor projections onto multilinear bases
  
  % Compute the convolutions with the filters for each timepoint
  for t1 = 1:szY(end) %TODO Can be done parallelly or on GPU 
    if opt.zeros_ignore % Treat zeros as missing data
      missing_filt = ones(opt.m,opt.m);
      if opt.W_force_round, [~,missing_filt] = transform_inds_circ(0,0,150,opt.m,(opt.m-1)/2,0); end
      conv_result_missing = conv2(data.proc_stack.Y(:,:,t1),missing_filt,'same');
    end
    for obj_type = 1:opt.NSS
      conv_result = zeros([szY(1:2), length(opt.Wblocks{obj_type})]); % Result of the convolution with each filter belonging to the current object
      if opt.standardise_cumulants && opt.mom>2, conv_result_standardised = conv_result; end % Use the pixelwise whitened data to compute when mom > 2 to get standardised cumulants
      for filt = opt.Wblocks{obj_type} %Convolution with each filter for an object type
        Wconv = imrotate(reshape(W(:,filt),opt.m,opt.m),180); % Need to rotate 180 degrees so it works correct with conv2
        conv_result(:,:,mod(filt-1,opt.KS)+1) = conv2(data.proc_stack.Y(:,:,t1),Wconv,'same');    
        if opt.standardise_cumulants && opt.mom>2 % Use the pixelwise whitened data to compute when mom > 2 to get standardised cumulants
          conv_result_standardised(:,:,mod(filt-1,opt.KS)+1) = conv2( ...
            (data.proc_stack.Y(:,:,t1)-opt.cumulant_pixelwise{1})./opt.cumulant_pixelwise{2},... % convolve standardised image
            Wconv,'same'); 
        end
        if opt.zeros_ignore % Multiply the conv result by number of missing elements
          conv_result(:,:,mod(filt-1,opt.KS)+1) = ...
            conv_result(:,:,mod(filt-1,opt.KS)+1)...
            .*(sum(missing_filt(:))./(sum(missing_filt(:))-conv_result_missing));
          cur_conv_result = conv_result(:,:,mod(filt-1,opt.KS)+1);
          % If too many zeros, just put zero as the convolution result
          cur_conv_result(...
            (conv_result_missing(:)./sum(missing_filt(:)))>(1-opt.zeros_min_nonzero_frac_needed)) ...
            = 0;
          conv_result(:,:,mod(filt-1,opt.KS)+1) = cur_conv_result;
          if opt.standardise_cumulants && opt.mom>2 % also fix for standised conv result
            cur_conv_result_standardised = conv_result_standardised(:,:,mod(filt-1,opt.KS)+1);
            cur_conv_result_standardised = cur_conv_result_standardised ...
              .*(sum(missing_filt(:))./(sum(missing_filt(:))-conv_result_missing));
            cur_conv_result_standardised(...
              (conv_result_missing(:)./sum(missing_filt(:)))>(1-opt.zeros_min_nonzero_frac_needed)) ...
              = 0;
            conv_result_standardised(:,:,mod(filt-1,opt.KS)+1) = cur_conv_result_standardised;
          end
        end
      end
      for mom1 = 1:opt.mom  %Get raw moments of the projected time course at each possible cell location %TODO - this might be wrong, because it assumes equal weighting??? but it's filter by filter, so maybe the linear combination of filters is still linear in the higher moments %TOTHINK Nah it seems correct
        [all_combs, mom_combs] = all_filter_combs(opt.KS, mom1, opt.diag_tensors); % Get the required tensors

        for i1 = 1:size(mom_combs,1) % Iterate over the possible combinations (should be opt.KS^mom1)
          % Turn the rows of all_combs into a vector that counts the occurance
          % of the k-th number for easy use with existing data_structure
          if opt.standardise_cumulants && mom1>2
            WY{obj_type}{mom1}(:,:,i1) = WY{obj_type}{mom1}(:,:,i1) + prod(bsxfun(@power, conv_result_standardised, shiftdim(mom_combs(i1,:),-1)),3);
          else
            WY{obj_type}{mom1}(:,:,i1) = WY{obj_type}{mom1}(:,:,i1) + prod(bsxfun(@power, conv_result, shiftdim(mom_combs(i1,:),-1)),3);
          end
        end
      end
    end
  end

  % Devide by number of timesteps to get the actual moment tensors
  for obj_type = 1:opt.NSS
    for mom1 = 1:opt.mom
      WY{obj_type}{mom1} = WY{obj_type}{mom1}./szY(3);
    end
  end

%   if opt.standardise_cumulants
%     warning('CHOMP:compute_filters(), Standardising multivariate cumulants is not implemented, using raw multivariate cumulants')
%   end
  
  %Convert the raw moment estimates into cumulant estimates
  for obj_type = 1:opt.NSS
    if opt.diag_tensors
      if ~opt.standardise_cumulants || opt.mom<=2
        WY{obj_type} = raw2cum(WY{obj_type});
      else  % If using standised cumulants for mom1 > 2, no need for raw2cum, already accounted for
        WY{obj_type}(1:2) = raw2cum(WY{obj_type}(1:2)); 
      end
    else
      % TODO - This is quite inefficient in terms of access, but for proof of
      % concept it's ok

      % Store the tensor-collapsing and tensor-inducing index vectors (takes long if within the loop)
      comb_inds_all = cell(opt.mom,2);
      for mom1 = 1:opt.mom
        [~, ~, comb_inds, comb_inds_rev] = all_filter_combs(opt.KS, mom1, opt.diag_tensors);
        comb_inds_all{mom1, 1} = comb_inds;
        comb_inds_all{mom1, 2} = comb_inds_rev;
      end

      for i1 = 1:szY(1)
        for i2 = 1:szY(2)
          % For every location, convert into a cell array with moments as its
          % elements as a proper tensor
          tmp = cell(opt.mom,1);
          for mom1 = 1:opt.mom
            tmp{mom1} = reshape(WY{obj_type}{mom1}(i1,i2,comb_inds_all{mom1,2}),[opt.KS*ones(1,mom1),1]);
          end

          % Compute the cumulants
          tmp = raw2cum_multivariate(tmp);
          if opt.standardise_cumulants
            % Unsure how to deal with this case
            error('CHOMP:compute_filters:opt.diag_tensors=0 and opt.standardise_cumulants=1 is not implemented, try different settings');
          end

          % Convert the resulting cumulant tensors back (into the cheap storeage of only unique elements) and store
          for mom1 = 1:opt.mom
            WY{obj_type}{mom1}(i1,i2,:) = tmp{mom1}(comb_inds_all{mom1,1});
          end
        end
      end  
    end
  end
  
  if opt.diag_cumulants_offdiagonly
    WY_full = WY;
  end
end
  
if opt.diag_cumulants==1 || opt.diag_cumulants_offdiagonly || opt.diag_cumulants_extradiagonal % use the projections of pixelwise cumulants onto the bases

  for obj_type = 1:opt.NSS
    for mom1 = 1:opt.mom
      [all_combs, mom_combs] = all_filter_combs(opt.KS, mom1, opt.diag_tensors);
      if opt.diag_cumulants_extradiagonal 
        if mom1 == 1
          continue; % We do not need to add extra filters for the mean
        end
        [all_combs, mom_combs] = all_filter_combs(opt.KS, mom1, 1); % Use diag tensors for this part for clarity
      end
      
      for i1 = 1:size(mom_combs,1) % Iterate over the possible combinations (should be opt.KS^mom1)
        % First calculate pointwise filter product combinations, then
        % convolve the pointwise pixel cumulant images with them
        Wcomb_cur = prod(bsxfun(@power, W(:,opt.Wblocks{obj_type}), mom_combs(i1,:)),2);
        
        % Turn it into a convolution filter
        Wconv_cur = rot90(reshape(Wcomb_cur,opt.m,opt.m),2); % Need to rotate 180 degrees so it works correct with conv2
        
        % Do the convolution of the cumulant image order with the combined
        % filter
        if opt.diag_cumulants_extradiagonal && mom1>=2 % Use diag tensors for this part for clarity
          WY{obj_type}{mom1}(:,:,end+1) = ... % keep adding the new extra diagonal projections simply here
            conv2(opt.cumulant_pixelwise{mom1},Wconv_cur,'same');
        else % just do normally
          WY{obj_type}{mom1}(:,:,i1) = conv2(opt.cumulant_pixelwise{mom1},Wconv_cur,'same');
        end
      end
    end
  end
end

if opt.diag_cumulants_offdiagonly % Subtract the diagcumulants from full cumulants
  for obj_type = 1:opt.NSS
    for mom1 = 1:opt.mom
      WY{obj_type}{mom1} = WY_full{obj_type}{mom1} - WY{obj_type}{mom1};
    end
  end
end

%% Computing WnormInv and GW - interactions between filters and projections

for obj_type = 1:opt.NSS
  for mom1 = 1:opt.mom  %Get raw moments of the projected time course at each possible cell location %TODO - this might be wrong, because it assumes equal weighting??? but it's filter by filter, so maybe the linear combination of filters is still linear in the higher moments %TOTHINK Nah it seems correct

    [all_combs, mom_combs, comb_inds, comb_inds_rev] = all_filter_combs(opt.KS, mom1, opt.diag_tensors);
    
    all_combs = all_combs(comb_inds,:); % Use only unique combinations - everything else is (super)symmetric - be careful to symmetrize during (re)constructions 
    if opt.diag_cumulants_extradiagonal && mom1>=2
       [extra_all_combs, ~, extra_comb_inds, ~] = all_filter_combs(opt.KS, mom1, 1);
       extra_all_combs = extra_all_combs(extra_comb_inds,:);
    else
      extra_all_combs=[];
    end

    
    % For each row in all_combs compute the filter 
    for filt1_ind = 1:(size(all_combs,1)+size(extra_all_combs,1))
      % Compute the filter tensor
      if filt1_ind <= size(all_combs,1)
        filt1 = get_filter_comb(W(:,opt.Wblocks{obj_type}), all_combs(filt1_ind,:), opt.diag_cumulants);
      else
        filt1_diag = get_filter_comb(W(:,opt.Wblocks{obj_type}), extra_all_combs(filt1_ind-size(all_combs,1),:), 1); %have diag_cumulants=1 for this
        % This returned the diagonal of whatever tensor I want, augment it
        % appropriately;
        filt1 = zeros((opt.m.^2).^mom1,1);
        filt1(linspace(1,size(filt1,1),opt.m^2)) = filt1_diag(:); % fill in the nonzeros efficiently;
      end
      for filt2_ind = filt1_ind:(size(all_combs,1)+size(extra_all_combs,1))
        if filt2_ind <= size(all_combs,1)
        filt2 = get_filter_comb(W(:,opt.Wblocks{obj_type}), all_combs(filt2_ind,:), opt.diag_cumulants);
        else
          filt2_diag = get_filter_comb(W(:,opt.Wblocks{obj_type}), extra_all_combs(filt2_ind-size(all_combs,1),:), 1); %have diag_cumulants=1 for this
          % This returned the diagonal of whatever tensor I want, augment it
          % appropriately;
          filt2 = zeros((opt.m.^2).^mom1,1);
          filt2(linspace(1,size(filt2,1),opt.m^2)) = filt2_diag(:); % fill in the nonzeros efficiently;
        end
        
        filt1 = filt1(:);
        filt2 = filt2(:);
        if opt.diag_cumulants_offdiagonly % Blank out the diagonals of the high dimensional Ws
          if mom1 > 1 % do this, otherwise this might mess up WnormInv{1}{1} with all zero filters
            filt1(linspace(1,numel(filt1),opt.m^2)) = 0;
            filt2(linspace(1,numel(filt2),opt.m^2)) = 0;
          end
        end
        
        % TOTHINK - Do we need to renormalise these objects? Think not. Old code did it though
        
        % Computing WnormInv - reconstruction update
        % ----------------------------------------
        % no extra filters here
        WnormInv{obj_type}{mom1}(filt1_ind,filt2_ind) = filt1(:)'*filt2(:);
        WnormInv{obj_type}{mom1}(filt2_ind,filt1_ind) = WnormInv{obj_type}{mom1}(filt1_ind,filt2_ind); %Use symmetricity
        
      end 
    end
    
    
    if ~opt.diag_tensors
      % But also this low dimensional WnormInv only computed for unique
      % combinations essentially sums together X values with repeated
      % combinations, we need to correct for this before reconstrcution
      comb_repeats = zeros(size(WnormInv{obj_type}{mom1},1),1);
      for comb1 = 1:length(comb_inds_rev)
        comb_repeats(comb_inds_rev(comb1)) = comb_repeats(comb_inds_rev(comb1)) + 1;
      end
      % Correct for these overcounting when computing the unique
      % reconstruction coefficients
      WnormInv{obj_type}{mom1} = diag(comb_repeats)*WnormInv{obj_type}{mom1}(1:numel(comb_repeats),1:numel(comb_repeats));
      if opt.diag_cumulants_extradiagonal
        % Currently unsure how to fix the likely resulting non-zero
        % offdiagonal elements in WnormInv, so throw error for now
        error('CHOMP:compute_filters(), Not implemented to have opt.diag_tensors=0, and also opt.diag_cumulants_extradiagonal=1. Change one of these settings!');
      end
    end
    

    
    if ~isempty(opt.W_weights)
      % Penalise the use of each basis function based on its inverse singular
      % value (stored in opt.W_weights already, as result of update_dict() )
    
      if opt.diag_cumulants_extradiagonal
        error('CHOMP:compute_filters(), Not implemented to have penalised regression with opt.W_weights non-empty, and also opt.diag_cumulants_extradiagonal=1. Change one of these settings!');
      end
      
      if opt.diag_tensors 
        % Simply add the squared inverse weights as diagonal matrix 
        % (solving the lin reg penalised by lambda*||opt.W_weights*x||^2),
        % where lambda = the mean of the diagonal of W'W (essentially average norm
        % of the bases)
        WnormInv{obj_type}{mom1} = WnormInv{obj_type}{mom1} + ...
          mean(diag(WnormInv{obj_type}{mom1}))*diag(opt.W_weights.^(mom1*2));
      else
        % Here we need to again calculate how many of which bases are being
        % activated, and penalise accordingly
        % need to compute W_weights * mom_combs appropriately
        % No time for now, throw error and implement later
        error('CHOMP:compute_filters(), Not implemented to have penalised regression with opt.W_weights non-empty, and also opt.diag_tensors=1. Change one of these settings!');
      end
    else % simple regularisation
      reg = 1e-6 * eye(size(WnormInv{obj_type}{mom1})); % Simple regularisation
      WnormInv{obj_type}{mom1} = WnormInv{obj_type}{mom1} + mean(diag(WnormInv{obj_type}{mom1}))*reg;
    end
    
    % Compute and store the inverse (so far WnormInv really was Wnorm...)  
    WnormInv{obj_type}{mom1} = inv(WnormInv{obj_type}{mom1});
      
  end
end
  



%% Computing GW - reconstruction update

% GW IS NOT USED ANYMORE, UMAP IS COMPUTED INSTEAD IN EXTRACT_COEFS
% ----------------------------------------
% Each cell is going to be a cell of (2*m-1)^2 shifts and at each shift and
% each moment we'll have a vector of features^moment to describe how much
% the corresponding WY entry is modified if we set the coeffecient of active
% filt1 at moment mom to 1.
%
% We need to compute the interaction between all filter responses,
% regardless of objecet type, so we need an extra loop for filt2 in
% here, and index accordingly

% for mom1 = 1:opt.mom
%   [all_combs, ~, comb_inds] = all_filter_combs(opt.KS, mom1, opt.diag_tensors);
% 
%   all_combs = all_combs(comb_inds,:); % Use only unique combinations - everything else is (super)symmetric - be careful to symmetrize during (re)constructions 
%   szAC = size(all_combs);
%       
%   for filt1_ind = 1:size(GW{mom1},1)
%     obj_type1 = floor((filt1_ind-1)/szAC(1))+1;
%     % Compute the filter tensor
%     filt1 = get_filter_comb(W(:,opt.Wblocks{obj_type1}), all_combs(mod(filt1_ind-1,szAC(1))+1,:));
%     for filt2_ind = filt1_ind:size(GW{mom1},1)
%       obj_type2 = floor((filt2_ind-1)/szAC(1))+1;
%       filt2 = get_filter_comb(W(:,opt.Wblocks{obj_type2}), all_combs(mod(filt2_ind-1,szAC(1))+1,:));
%         
%       if 0==0 % exist(['./Subfuncs/Compute/Mex/computeGW.' mexext],'file') %Quicker c for loop
%         GW{mom1}{filt1_ind,filt2_ind} = computeGW(GPT,filt1,filt2,opt.m,opt.mom,mom1);
%         GW{mom1}{filt2_ind,filt1_ind} = rot90(GW{mom1}{filt1_ind,filt2_ind},2); % See explaination in Matlab version
%       else %Slower Matlab for loop
%         GW{mom1}{filt1_ind,filt2_ind} = zeros(2*opt.m-1, 2*opt.m-1); 
%         for s1 = 1:(2*opt.m-1)
%           for s2 = 1:(2*opt.m-1)
%             GW{mom1}{filt1_ind,filt2_ind}(s1,s2) = filt1(GPT{s1,s2,mom1}(:,2))'* filt2(GPT{s1,s2,mom1}(:,1)); %compute the shifted effect in original space via the shift tensors GPT. Because the Worigs were computed to correspond to the best inverse of the Ws
%             GW{mom1}{filt2_ind,filt1_ind}(2*opt.m-s1,2*opt.m-s2) = GW{mom1}{filt1_ind,filt2_ind}(s1,s2); % Use symmetricity (Swapping filter indices causes the shift to be reflected around the (m,m) point as origin, when we swap filters thus s -> 2m-s
%           end
%         end
%       end
%     end
%   end
% end

clearvars -except WY GW WnormInv







end

