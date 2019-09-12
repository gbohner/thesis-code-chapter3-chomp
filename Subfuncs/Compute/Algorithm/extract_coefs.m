function [ H, X, L] = extract_coefs( WY, GW, WnormInv, W, opt, varargin)
%EXTRACT_COV_COEFS Summary of this function goes here
%   Detailed explanation goes here

% opt.fig = 1; %TMP
if opt.fig > 1
%   h_dl = figure(7);  
%   h_dl2 = figure(8);  
%   h_dl3 = figure(9);
  if opt.fig > 2
    h_comb = figure(107);
    set(h_comb,'Units','Normalized')
    set(h_comb,'Position',[0.1,0.1,0.7,0.8]);
  end
%   if opt.fig >3
% %     Video_dl = VideoWriter('likelihood_map.avi');
% %     Video_dl.FrameRate = 2;
% %     open(Video_dl);
% %     Video_dl2 = VideoWriter('mean_score_map.avi');
% %     Video_dl2.FrameRate = 2;
% %     open(Video_dl2);
% %     Video_dl3 = VideoWriter('var_score_map.avi');
% %     Video_dl3.FrameRate = 2;
% %     open(Video_dl3);
%     Video_comb = VideoWriter('inference_video.avi');
%     Video_comb.FrameRate = 2;
%     open(Video_comb);
%   end
end

Ntypes = opt.NSS;
szWY = size(WY{1}{1}); % DataWidth * DataHeight * n_filter

n_regressors = zeros(opt.mom,1);
for mom1 = 1:opt.mom
  n_regressors(mom1) = size(WY{1}{mom1},3);
end

H = zeros(opt.cells_per_image,3); %Location (row, col, type)
X = zeros(opt.cells_per_image, sum(n_regressors)); % Basis function coefficients
L = zeros(opt.cells_per_image, opt.mom); % Likelihood gains


if opt.mask
  Mask = varargin{1};
else
  Mask = ones(szWY(1:2)); % Possible cell placements (no overlap / nearby cells);
end

Mask(1:floor(opt.m/4),:) = 0; Mask(end-floor(opt.m/4):end,:) = 0; Mask(:, 1:floor(opt.m/4)) = 0; Mask(:, end-floor(opt.m/4):end) = 0;%Don't allow cells near edges
%Mask(1:floor(opt.m),:) = 0; Mask(end-floor(opt.m):end,:) = 0; Mask(:, 1:floor(opt.m)) = 0; Mask(:, end-floor(opt.m):end) = 0;%Don't allow cells near edges
Mask = double(Mask);
%TODO: Maybe modify it such that mask is per object type

%% Initialize coefficients and likelihood maps

WYorig = WY; % Store original coefficients to keep track of changes


dL_mom = zeros([szWY(1:2),Ntypes,opt.mom]);
if opt.W_addflat
  dL_mom_flat = zeros([szWY(1:2),Ntypes,opt.mom]);
end
dL = zeros([szWY(1:2), Ntypes]); % delta log likelihood
xk = cell(opt.NSS,1); % Coefficients for image filter reconstruction
for obj_type=1:opt.NSS
  xk{obj_type} = cell(opt.mom,1);
%   for mom1 = 1:opt.mom
%     xk{obj_type}{mom1} = zeros(size(WY{obj_type}{mom1}));
%   end
end


if opt.reconst_upto_median_WY
  % Get each WY median for each moment and use WnormInv to uncorrelate
  % those, then subtract from WY to reconst up to median ("noise")
  
  for obj_type=1:opt.NSS
    for mom1 = 1:opt.mom
      cur_WYmedians = zeros(size(WY{obj_type}{mom1},3),1);
      for filt_ind = 1:length(cur_WYmedians)
        cur_WYmedians(filt_ind) = get_hist_mode(WY{obj_type}{mom1}(:,:,filt_ind),2000,0);
        % Subtract from WY
        WY{obj_type}{mom1}(:,:,filt_ind) = ...
          WY{obj_type}{mom1}(:,:,filt_ind) - cur_WYmedians(filt_ind);
      end
      
%       
%       cur_WYmedians = WnormInv{obj_type}{mom1}*cur_WYmedians;  This is NOT necessary! It is effectively done later
%       for filt_ind = 1:length(cur_WYmedians)
%         WY{obj_type}{mom1}(:,:,filt_ind) = ...
%           WY{obj_type}{mom1}(:,:,filt_ind) - cur_WYmedians(filt_ind);
%       end
    end
  end
      


  % Otherwise use this expected-noise-based approach below that works OK with
  % the diagonals?
elseif ~strcmp(opt.learn_decomp, 'NMF') && ~opt.diag_cumulants_offdiagonly && ~opt.whiten
  % Only do this non-nmf style stuff, for NMF we want to reconstruct additively
  % Also, this subtracts diagonal cumulants, when we do offdiagonly, it is
  % wrong to do

  % For the mean and variance WY maps, compute the mode and subtract it from WY. That way we
  % ensure we reconstruct towards the mode (background mean and noise variance) rather than towards 0
  % Can also do this "mode" finding thing locally potentially, but after
  % preproc2P we don't need to
  %if there is mask image, ensure to reject those elements from the mode
  %calciulations
  
  for obj_type=1:opt.NSS
    W_weights = {};
    W_weights{1} = sum(W(:,opt.Wblocks{obj_type}),1)';
    W_weights{2} = W(:,opt.Wblocks{obj_type})'*W(:,opt.Wblocks{obj_type});
    [all_combs, mom_combs, comb_inds, comb_inds_rev] = all_filter_combs(opt.KS, 2, opt.diag_tensors);
    if opt.diag_tensors
      W_weights{2} = diag(W_weights{2});
    else
      W_weights{2} = reshape(W_weights{2},[],1);
      W_weights{2} = W_weights{2}(comb_inds); % Select only the unique weights corresponding to WY's last dim
    end
    if opt.diag_cumulants_extradiagonal
      if ~opt.diag_tensors
        error('Not implemented')
      end
      W_weights{2} = [W_weights{2}(:); W_weights{2}(:)];
    end


    for mom1 = 1:min(opt.mom,2+2*opt.diag_cumulants) % if diag_cumulants, we can do the reconstruction towards "mode" in every order
        tmp_true_modes{mom1} = get_hist_mode(opt.cumulant_pixelwise{mom1},500); %

        [all_combs, mom_combs, comb_inds, comb_inds_rev] = all_filter_combs(opt.KS, mom1, opt.diag_tensors);

        if opt.diag_cumulants
          cur_all_combs = all_combs(comb_inds,:);
          W_weights{mom1} = zeros(size(WY{obj_type}{mom1},3),1);
          for i111 = 1:size(cur_all_combs,1)
            cur_comb_W = get_filter_comb(W(:,opt.Wblocks{obj_type}), cur_all_combs(i111,:), opt.diag_cumulants);
            W_weights{mom1}(i111) = sum(cur_comb_W(:));
          end
        end

        % Subselect dims that have non-orthogonal basis combinations
        % (orthogonal bases should reject the noise?)
        tmp_nonorth_b = sum((mom_combs(comb_inds_rev,:)==mom1),2);
        tmp_nonorth_b = logical(tmp_nonorth_b(comb_inds)); % Select non-orthogonal bases combinations that keep noise
        tmp_nonorth_b(abs(W_weights{mom1}(tmp_nonorth_b))<0.5) = 0; % Reject low weights from backprojection


        if sum(tmp_nonorth_b) == 0 % if all bases reject noise, we're ok
          %fprintf('All order %d bases sum to zero!%n', mom1);
          tmp_mode_approx = nan;
          % Find the mode of backprojected WYs (this could be done earlier I
          % guess?) % Reweight by basis weights before finding the mode
        else
          tmp_mode_approx = get_hist_mode(WY{obj_type}{mom1}(:,:,tmp_nonorth_b).*reshape(1./W_weights{mom1}(tmp_nonorth_b),1,1,[]), 1300);
        end

        % Subtract the weighted mode from each dimension (as if that mode
        % projected into the appropriate basis)
        tmp_all_filter_inds = 1:size(WY{obj_type}{mom1},3);
        for cur_filt_comb = tmp_all_filter_inds%(tmp_nonorth_b)
          WY{obj_type}{mom1}(:,:,cur_filt_comb) = ... % Use the mode estimated from mean and variance images, the backprojection thing doesn't work for some reason?
            WY{obj_type}{mom1}(:,:,cur_filt_comb) - tmp_true_modes{mom1}*W_weights{mom1}(cur_filt_comb); % Remove noise power from affected dimensions
            %WY{obj_type}{mom1}(:,:,cur_filt_comb) - tmp_mode*W_weights{mom1}(cur_filt_comb); % Remove noise power from affected dimensions
        end
        fprintf('\nSubtracted WY{%d}{%d} mode of %0.2e, backprojected approx is %0.2e ...', obj_type, mom1, tmp_true_modes{mom1}, tmp_mode_approx);
    end
  end
  
end




%% Locate cells 

if opt.local_peaks
  if isempty(opt.local_peaks_size)
    opt.local_peaks_size = [opt.m, (opt.m-1)/2,0];
  end
  [~, context_conv_filter] = transform_inds_circ(0,0,150,opt.local_peaks_size(1),opt.local_peaks_size(2),opt.local_peaks_size(3));
  if opt.local_peaks_gauss >0
    tmp_dt = -((opt.local_peaks_size(1)-1)/2):((opt.local_peaks_size(1)-1)/2);
    tmp_dt = tmp_dt(:);
    tmp_gauss_filter = exp(-tmp_dt.^2/(2*opt.local_peaks_gauss.^2)) * exp(-tmp_dt'.^2/(2*opt.local_peaks_gauss.^2));
    context_conv_filter = context_conv_filter.*tmp_gauss_filter;
  end
  local_num_elem = conv2(ones(szWY(1),szWY(2)), context_conv_filter, 'same');
end

for j = 1:opt.cells_per_image
  
  % Compute delta log-likelihoods
  if j == 1
    % Compute filter coefficients (MAP estimate)  - we'll update it in small areas
    for obj_type=1:opt.NSS
      for mom1 = 1:opt.mom
        xk{obj_type}{mom1} = ...
          mply(WY{obj_type}{mom1}, WnormInv{obj_type}{mom1}, 1);
        
        if strcmp(opt.learn_decomp, 'NMF')
          xk{obj_type}{mom1}(xk{obj_type}{mom1}<0) = 0; % Project onto positive only space
        end
       
      end
    end
  end
  
  
  %Compute delta log-likelihood
  for obj_type=1:opt.NSS
    for mom1 = 1:opt.mom
        if ~opt.W_addflat
          dL_mom(:,:,obj_type,mom1) = ...
            -sum(WY{obj_type}{mom1} .* xk{obj_type}{mom1},3);
        else % the reconstruction with respect to the very first W shouldn't matter
          dL_mom(:,:,obj_type,mom1) = ...
            (-sum(WY{obj_type}{mom1}(:,:,2:end) .* xk{obj_type}{mom1}(:,:,2:end),3)) ... % gain due to structured part
            - (-sum(WY{obj_type}{mom1}(:,:,1) .* xk{obj_type}{mom1}(:,:,1),3)); % gain due to unstructured part
        end
    end
  end
  
  % Renormalise
  dL_mom_orig = dL_mom;
  for obj_type=1:opt.NSS
    mom_weights = ones(opt.mom,1);
    for mom1 = 1:opt.mom
        % If spatial gain is supplied, use it to multiply expected likelihood
        % changes appropriately (we originally divided by this, so
        % multiplication with .^(2*mom1) undoes it effectively; likelihood change is squaring the effect of spatial_gain).
        if ~isempty(opt.spatial_gain) && opt.spatial_gain_renormalise_lik
          if ~opt.standardise_cumulants || mom1 <= 2 % standardised cumulants > 2 are not affected by spatial gain!
            if opt.spatial_gain_undo_raw % We multiplied the input data by spatial gain already, so no we need to divide back by it for weighting
              dL_mom(:,:,obj_type,mom1) = ...
                dL_mom(:,:,obj_type,mom1)...
                ./(opt.spatial_gain.^(2*mom1)); 
            else % we did not multiply the input data by it, and we want to undo its effects
              dL_mom(:,:,obj_type,mom1) = ...
                dL_mom(:,:,obj_type,mom1)...
                .*(opt.spatial_gain.^(2*mom1));                
            end
          end
        end
      
      
        % TOTHINK: Give relative weight to the moments based on how many elements
        %they involve
        
%         % Use the inverse norm as weighting for the summation;
%         mom_weights(mom1,1) = 1./norm(reshape(dL_mom(:,:,obj_type,mom1),1,[])); 
        
        % TOTHINK: compute the likelihood change not just based on how much
        % the likelihood improve, but also give a penalty for using certain
        % basis functions given their singular value during learning step
        % dL_mom(:,:,obj_type,mom) = - sum(WY(:,:,opt.Wblocks{type},mom) .* xk(:,:,opt.Wblocks{type},mom),3);
  %       if mom>=2
  %         dL_mom(:,:,mom) = dL_mom(:,:,mom)./abs(mean2(dL_mom(:,:,mom))); %normalize the moment-related discrepencies
  %       end
       if ~opt.local_peaks % Do global z-scoring
         szdL_mom = [size(dL_mom), 1, 1];
         dL_mom = reshape(dL_mom, [prod(szdL_mom(1:2)), szdL_mom(3), szdL_mom(4)]);
         dL_mom(Mask(:)>0,obj_type,mom1) = ...
           zscore(dL_mom(Mask(:)>0,obj_type,mom1),[], 1);
         dL_mom = reshape(dL_mom, szdL_mom);
       else  % do local convolutional z-scoring ("context normalisation")
         context_mean = conv2(dL_mom(:,:,obj_type,mom1), context_conv_filter, 'same')./local_num_elem;
         context_mom2 = conv2(dL_mom(:,:,obj_type,mom1).^2, context_conv_filter, 'same')./local_num_elem;
         context_std = sqrt(context_mom2-context_mean.^2);
         dL_mom(:,:,obj_type,mom1) = ...
           (dL_mom(:,:,obj_type,mom1) - context_mean)./ context_std;
       end
         
    end
    
    % If the user supplied extra weighting, apply that as well:
    if ~isempty(opt.mom_weights)
      mom_weights = mom_weights(:) .* opt.mom_weights(:);
    end
    
    %linear sum of individual zscored differences %TODO GMM version of "zscoring jointly"

    dL(:,:,obj_type) = mply(dL_mom(:,:,obj_type,:), mom_weights, 1); 
%     dL = dL_mom(:,:,obj_type,4); % Make it kurtosis pursuit
  end

  if mod(j,20)==2
    j111=j; %disp(j);
  end
  % Find maximum decrease  
  [AbsMin, ind] = min( dL(:).*repmat(Mask(:),Ntypes,1) );
  [row_hat, col_hat, type_hat] = ind2sub(size(dL),ind); 
    
  %fprintf('%0.2f', AbsMin);
  %if mod(j,20), fprintf('\n'),end
  %Check if there is not enough likelihood decrease anymore
  if AbsMin >= 0
    H = H(1:(j-1), :);
    break;
  end    
  
  %Store the values
  H(j, :) = [row_hat, col_hat, type_hat]; % Estimated location and type
  for mom1 = 1:opt.mom
    X(j, sum([0; n_regressors(1:mom1-1)])+1:sum(n_regressors(1:mom1))) = ...
      squeeze(xk{type_hat}{mom1}(row_hat,col_hat,:));
  end
  L(j,:) = squeeze(dL_mom(row_hat,col_hat,type_hat,:)); % Estimated likelihood gain per moment
    
    
  if opt.fig >1
%     set(0,'CurrentFigure',h_dl); imagesc(dL_mom(:,:,1)); colorbar; pause(0.05);
%     set(0,'CurrentFigure',h_dl); imagesc(dL(:,:,1).*Mask); colorbar; axis square;  pause(0.05);
%     set(0,'CurrentFigure',h_dl2); imagesc(dL_mom(:,:,1,min(1,size(dL_mom,4))).*Mask); colorbar; axis square; pause(0.05);
%     set(0,'CurrentFigure',h_dl3); imagesc(dL_mom(:,:,1,min(2,size(dL_mom,4))).*Mask); colorbar; axis square; pause(0.05);
  
    if opt.fig >2
      
      
      
%       [~,rl,~] = reconstruct_cell( opt, W, X(1:j,:));
%       
%       full_reconst = zeros(szWY(1:2));
%       for c1 = 1:size(rl{1},3)
%         row_hat = H(c1,1); col_hat = H(c1,2);
%         [inds, cut] = mat_boundary(size(full_reconst),row_hat-floor(opt.m/2):row_hat+floor(opt.m/2),col_hat-floor(opt.m/2):col_hat+floor(opt.m/2));
%         full_reconst(inds{1},inds{2}) = full_reconst(inds{1},inds{2}) + rl{1}(1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2),c1);
%       end
%      %load(get_path(opt,'output_iter',4),'model')
%      %set(0,'CurrentFigure',h_comb);
%      %subplot(2,2,1); imagesc(model.y_orig); colormap gray; axis square; axis off; title('Mean Data');
%      subplot(2,2,1); imagesc(-dL(:,:,1)); colormap default; colorbar; axis square; axis off; title('Log likelihood increase');  pause(0.0001);
%      subplot(2,2,3); imagesc(full_reconst); colorbar; axis square; axis off; title('Reconstructed cells'); pause(0.0001); 
%      subplot(2,2,2); imagesc(-dL_mom(:,:,min(1,size(dL_mom,3)),min(1,size(dL_mom,4)))); colorbar; axis square; axis off; title('mom=1 LL change'); pause(0.0001);
%      subplot(2,2,4); imagesc(-dL_mom(:,:,min(1,size(dL_mom,3)),min(4,size(dL_mom,4)))); colorbar; axis square; axis off; title(sprintf('mom=%s LL change',size(dL_mom,4))); pause(0.0005);
%    set(0,'CurrentFigure',h_dl3); imagesc(Mask(:,:,1)); colorbar; axis square; pause(0.05);
    end
  end
  
  
  
  
  %Affected local area
  % Size(,1) : number of rows, size(,2): number of columns
 [inds, cut] = mat_boundary(szWY(1:2),(row_hat-opt.m+1):(row_hat+opt.m-1),(col_hat-opt.m+1):(col_hat+opt.m-1));
  
 
 %% Update WY
 % Compute reconstruction
 if j==1 % Generate Wfull and the corresponding U linear update map for each object type
   load(get_path(opt, 'precomputed'), 'GPT');
   Wfull = cell(opt.NSS,1);
   Umap = cell(opt.NSS,opt.NSS,1);
   for obj_type = 1:opt.NSS
     % Get the high dimensional W tensors (B^r matrix)
     [~,~,Wfull_cur] = reconstruct_cell( opt, W(:,opt.Wblocks{obj_type}), X(j,:), 'do_subset_combs', true, 'do_reconstruction', false);
     Wfull{obj_type} = Wfull_cur;
   end
   
   if opt.blank_reconstructed
     blanked_areas_mask = ones(szWY(1),szWY(2)); % Keep track of which areas are blanked already, and blank parts of patches accordingly
     
     % Umap = cell(opt.NSS,1); % we don't need to know which object type we are mapping from, as we map from raw data
     % Create a Wfull-like object that is all ones in the reconstructed
     % area
     tmp_opt = struct_merge(chomp_options(), opt);
     tmp_opt.NSS = 1;
     tmp_opt.KS = 1;
     if ~opt.W_force_round % Overlaps are in square patches
       tmp_W = ones(size(W,1),1); % Do I want to normalise this??? Think not...
     else % overlaps only in circular areas
       [~, tmp_W] = transform_inds_circ(0,0,150,opt.m,(opt.m-1)/2,0);
       tmp_W = tmp_W(:);
     end
        
     tmp_X = ones(1,opt.mom); % Don't think this matters at all, but this is correct for NSS=1 and KS=1;
     [~, ~, Wallones] = reconstruct_cell( tmp_opt, tmp_W, tmp_X, 'do_subset_combs', false, 'do_reconstruction', false);
   
     Umap_numoverlap = Umap;
   end

   
   for obj_type_from = 1:opt.NSS
     for obj_type_to = 1:opt.NSS

       % Compute the linear update map Umap using Wfull and the precomputed
       % GPT shift tensor
       Umap{obj_type_to}{obj_type_from} = cell(opt.mom,1);
       if opt.blank_reconstructed
         Umap_numoverlap{obj_type_to}{obj_type_from} = cell(opt.mom,1);
       end
       for mom1 = 1:opt.mom
         % Get the partial unique mappings
         [all_combs, ~, comb_inds, comb_inds_rev] = all_filter_combs(opt.KS, mom1, opt.diag_tensors);
         
         % Calculate comb_repeats (each reconstruction and mapping should
         % be weighted by the number of times it contributes to the
         % higher order symmetric tensor)
         % Only use this for the reconst weights, not the shifted ones,
         % as WY stores single repeats (WnormInv takes care of finding the
         % correct X by "repeating" WY appropriately).
         comb_repeats = zeros(numel(comb_inds),1);
         for comb1 = 1:length(comb_inds_rev)
          comb_repeats(comb_inds_rev(comb1)) = comb_repeats(comb_inds_rev(comb1)) + 1;
         end
         
         if opt.diag_cumulants_extradiagonal
           % Make sure to extend the iterations (and thus Umap) to the
           % extra diagonal parts
           comb_repeats = [comb_repeats(:); ones(size(Wfull{obj_type_from}{mom1},2)-numel(comb_inds),1)]; % diag_tensors is true, so just 1 weights
           comb_inds = [comb_inds(:), ones(size(Wfull{obj_type_from}{mom1},2)-numel(comb_inds),1)];
         end
         
         if ~opt.blank_reconstructed % Learn the reconst_coefs -> WYchange mapping as Umap
           
           Umap{obj_type_to}{obj_type_from}{mom1} = zeros(2*opt.m-1,2*opt.m-1, numel(comb_inds), numel(comb_inds)); % P x unique_combs x unique_combs
         
           for filt_reconst_comb_ind = 1:numel(comb_inds) % Iterate through each unique filter combination 
             for filt_shifted_comb_ind = 1:numel(comb_inds) % Iterate through each unique filter combination
               for s1 = 1:2*opt.m-1
                 for s2 = 1:2*opt.m-1
                   if ~opt.diag_cumulants % use mom1-th order shifts
                     Umap{obj_type_to}{obj_type_from}{mom1}(s1,s2,filt_shifted_comb_ind, filt_reconst_comb_ind) = ... % Define the full %K^r -> unique_combs update mapping 
                       comb_repeats(filt_reconst_comb_ind)*... % this part computes how much each filter combination contributes to the full reconstructed tensor
                       Wfull{obj_type_to}{mom1}(GPT{s1,s2,mom1}(:,1), filt_shifted_comb_ind)'*Wfull{obj_type_from}{mom1}(GPT{s1,s2,mom1}(:,2),filt_reconst_comb_ind);
                   else % use the first order shifts only
                     Umap{obj_type_to}{obj_type_from}{mom1}(s1,s2,filt_shifted_comb_ind, filt_reconst_comb_ind) = ... % Define the full %K^r -> unique_combs update mapping 
                       comb_repeats(filt_reconst_comb_ind)*... % this part computes how much each filter combination contributes to the full tensor
                       Wfull{obj_type_to}{mom1}(GPT{s1,s2,1}(:,1), filt_shifted_comb_ind)'*Wfull{obj_type_from}{mom1}(GPT{s1,s2,1}(:,2),filt_reconst_comb_ind);
                   end
                 end
               end
             end
           end
           
         else % Learn the n_order_patch -> WYchange mapping as Umap, and also the number of elements as Umap_elems
           
           Umap{obj_type_to}{obj_type_from}{mom1} = zeros(2*opt.m-1,2*opt.m-1, numel(comb_inds), numel(Wallones{mom1})); % P x unique_combs x all_combs           
           %W_blanked_norm_corr = 
           
           for filt_shifted_comb_ind = 1:numel(comb_inds) % Iterate through only the unique filter combinations
             for s1 = 1:2*opt.m-1
               for s2 = 1:2*opt.m-1
                 
                   % At each shift we need to also compute the "blanked" W
                   % norms for proper correction
                   
                 
                 if ~opt.diag_cumulants % use mom1-th order shifts
                   Umap{obj_type_to}{obj_type_from}{mom1}(s1,s2,filt_shifted_comb_ind, GPT{s1,s2,mom1}(:,2)) = ... % Define the full %K^r -> unique_combs update mapping 
                    Wfull{obj_type_to}{mom1}(GPT{s1,s2,mom1}(:,1), filt_shifted_comb_ind).*Wallones{mom1}(GPT{s1,s2,mom1}(:,2)); % Map from each index in the n-th order all ones patch
                 else % use the first order shifts only
                   Umap{obj_type_to}{obj_type_from}{mom1}(s1,s2,filt_shifted_comb_ind, GPT{s1,s2,1}(:,2)) = ... % Define the full %K^r -> unique_combs update mapping 
                     Wfull{obj_type_to}{mom1}(GPT{s1,s2,1}(:,1), filt_shifted_comb_ind).*Wallones{mom1}(GPT{s1,s2,1}(:,2)); % Map from each index in the n-th order all ones patch
                 end
               end
             end
           end
           
           % Get the number of overlapping tensor elements (that now will
           % be treated as missing data);
           Umap_numoverlap{obj_type_to}{obj_type_from}{mom1} = zeros(2*opt.m-1,2*opt.m-1);
           for s1 = 1:2*opt.m-1
             for s2 = 1:2*opt.m-1
               if ~opt.diag_cumulants % use mom1-th order shifts
                 Umap_numoverlap{obj_type_to}{obj_type_from}{mom1}(s1,s2) = ... 
                  Wallones{mom1}(GPT{s1,s2,mom1}(:,1))'*Wallones{mom1}(GPT{s1,s2,mom1}(:,2)); 
               else % use the first order shifts only
                 Umap_numoverlap{obj_type_to}{obj_type_from}{mom1}(s1,s2) = ... 
                  Wallones{mom1}(GPT{s1,s2,1}(:,1))'*Wallones{mom1}(GPT{s1,s2,1}(:,2)); 
               end
             end
           end
           
           
           
         end

      end
     end
   end
 end
 
  %% Do the j-th cell's reconstruction
  if opt.blank_reconstructed
    % Get the appropriate nth order patches
    [cur_patch_inds, cur_patch_cut] = mat_boundary(szWY(1:2),(row_hat-(opt.m-1)/2):(row_hat+(opt.m-1)/2),(col_hat-(opt.m-1)/2):(col_hat+(opt.m-1)/2));
    
    if opt.diag_cumulants
      cur_cumulants = opt.cumulant_pixelwise;
      % Simple case, just use the patches from opt.cumulant_pixelwise
      cur_n_order_patch = cell(opt.mom,1);
      for mom1 = 1:opt.mom
        cur_n_order_patch{mom1} = zeros(opt.m,opt.m);
        cur_n_order_patch{mom1}((1+cur_patch_cut(1,1)):(end-cur_patch_cut(1,2)),(1+cur_patch_cut(2,1)):(end-cur_patch_cut(2,2))) = ...
         cur_cumulants{mom1}(cur_patch_inds{1},cur_patch_inds{2});

        if opt.W_force_round % blank out parts of the patch (Umap should already do this, but just in case)
          cur_n_order_patch{mom1} = cur_n_order_patch{mom1}.*reshape(tmp_W,opt.m,opt.m);
        end
        
        % Blank out previously used parts
        cur_n_order_patch{mom1}((1+cur_patch_cut(1,1)):(end-cur_patch_cut(1,2)),(1+cur_patch_cut(2,1)):(end-cur_patch_cut(2,2))) = ...
          cur_n_order_patch{mom1}((1+cur_patch_cut(1,1)):(end-cur_patch_cut(1,2)),(1+cur_patch_cut(2,1)):(end-cur_patch_cut(2,2))) ...
          .*blanked_areas_mask(cur_patch_inds{1},cur_patch_inds{2});
        
      end
    else % We need the full n-th order patch
      if j == 1
        % Access the data, we need to get the patches
        cur_inp = load(get_path(opt)); % Instead of this, later just fix so that we pass the dataset already in memory from Model_learn as varargin
        szY = chomp_size(cur_inp.inp.data.proc_stack,'Y');
      end
      % Get patch
      patches = get_patch(cur_inp.inp.data.proc_stack, opt, sub2ind([szY(1:2) opt.NSS], H(j,1), H(j,2), H(j,3)));

  %     if opt.W_addflat
  %       patches = bsxfun(@minus, patches, mean(mean(patches,2),1));
  %     end

      if opt.W_force_round % Blank corners if needed
        [~, tmp_cur_mask] = transform_inds_circ(0,0,150,opt.m,(opt.m-1)/2,0);
        patches = patches.*tmp_cur_mask;
      end
      
      % Blank out previously used parts
      patches((1+cur_patch_cut(1,1)):(end-cur_patch_cut(1,2)),(1+cur_patch_cut(2,1)):(end-cur_patch_cut(2,2)),:) = ...
        patches((1+cur_patch_cut(1,1)):(end-cur_patch_cut(1,2)),(1+cur_patch_cut(2,1)):(end-cur_patch_cut(2,2)),:) ...
        .*blanked_areas_mask(cur_patch_inds{1},cur_patch_inds{2});

      % Get n-order-patch
      cur_n_order_patch = get_n_order_patch(patches(:,:,:,1), opt, szY);
    end

    % Subtract the mode of the distribution if we normally do so
    if exist('tmp_true_modes','var')
      % Subtract the mode from each cumulant order (ie we dont reconstruct
      % noise)
      for mom1 = 1:opt.mom
        % Only non-zero elements (in case things were blanked out before)
        cur_n_order_patch{mom1}(cur_n_order_patch{mom1}~=0) = ...
          cur_n_order_patch{mom1}(cur_n_order_patch{mom1}~=0) - tmp_true_modes{mom1};
      end
    end
    
    % When we get a patch, blank it's place so blanking affects future
    % get_patch() calls!!!
    if ~opt.W_force_round
      blanked_areas_mask(cur_patch_inds{1},cur_patch_inds{2}) = 0;
    else
      tmp_W_reshaped = reshape(tmp_W,opt.m,opt.m);
      % Blank only the parts where tmp_W_reshape is non-zero (the central
      % circle)
      blanked_areas_mask(cur_patch_inds{1},cur_patch_inds{2}) = ...
        blanked_areas_mask(cur_patch_inds{1},cur_patch_inds{2}).*(0==tmp_W_reshaped((1+cur_patch_cut(1,1)):(end-cur_patch_cut(1,2)),(1+cur_patch_cut(2,1)):(end-cur_patch_cut(2,2))));
    end
  end

 WYchange = cell(opt.NSS,1);
 for obj_type = 1:opt.NSS
   WYchange{obj_type} = cell(opt.mom,1);
   
   Xinds_start = 0;
   
   for mom1 = 1:opt.mom
%      reconst_cur = reshape(reconst{mom1},opt.m, opt.m, []); % m x m x REST tensor

     %WYchange{obj_type}{mom1} = zeros(2*opt.m-1,2*opt.m-1,size(WY{obj_type}{mom1},3));
     
     % Get the partial unique mappings
     [all_combs, ~, comb_inds, comb_inds_rev] = all_filter_combs(opt.KS, mom1, opt.diag_tensors);
     
     if ~opt.blank_reconstructed % Use Umap to map the reconst to WYchange
       % Extend X(j,:) to appropriate K^r size from just unique combs (copied from reconstruct_cell)
       Xcur = X(j,(Xinds_start+1):(Xinds_start+size(WY{obj_type}{mom1},3))); % Extract relevant X weights for mom1
       Xinds_start = Xinds_start + size(WY{obj_type}{mom1},3);
       %Xcur = Xcur(:,comb_inds_rev); % no need to do this, since Umap maps from unique->unique, weighted by comb_repeats, see up there 

       WYchange{obj_type}{mom1} = mply(Umap{obj_type}{type_hat}{mom1}, Xcur(:), 1); % U update maps from K^r coeffs to P patch x the unique combs

      % Update WY locally
       WY{obj_type}{mom1}(inds{1},inds{2},:) = ...
         WY{obj_type}{mom1}(inds{1},inds{2},:) - WYchange{obj_type}{mom1}((1+cut(1,1)):(end-cut(1,2)),(1+cut(2,1)):(end-cut(2,2)),:);
     
     else % Treat the reconstructed patch as missing data, and update WY projections accordingly
       
       % Get n-tensors from data (load data earlier) via get_patches (at
       % current H) then get_n_order_patch
       
       WYchange{obj_type}{mom1} = mply(Umap{obj_type}{type_hat}{mom1}, cur_n_order_patch{mom1}(:), 1);
       
       % This WY_change is from the raw data patch, but we might have
       % projected already from that patch, take that into account
       cur_WYchange = WYchange{obj_type}{mom1}((1+cut(1,1)):(end-cut(1,2)),(1+cut(2,1)):(end-cut(2,2)),:);
       cur_WYorig = WYorig{obj_type}{mom1}(inds{1},inds{2},:);
       cur_WY = WY{obj_type}{mom1}(inds{1},inds{2},:);
       
       
       % Do the subtraction of all projections that were in the blanked
       % area
       WY{obj_type}{mom1}(inds{1},inds{2},:) = ...
         WY{obj_type}{mom1}(inds{1},inds{2},:) - WYchange{obj_type}{mom1}((1+cut(1,1)):(end-cut(1,2)),(1+cut(2,1)):(end-cut(2,2)),:);
       
%        % But after also do renormalisation due to less elements
%        tmp_total_elem = sum(Wallones{mom1}); % All non-zero elements in Wallones{mom1}
%        tmp_missing_elem = Umap_numoverlap{obj_type_to}{obj_type_from}{mom1};
%        tmp_div_by_normalise = ((tmp_total_elem-tmp_missing_elem)*1.)./(tmp_total_elem*1.); % in the very middle this is zero!!!
%        tmp_div_by_normalise(tmp_div_by_normalise<=0) = 1;
%        
%        % Do the division;
%        WY{obj_type}{mom1}(inds{1},inds{2},:) = ...
%          WY{obj_type}{mom1}(inds{1},inds{2},:)./ tmp_div_by_normalise((1+cut(1,1)):(end-cut(1,2)),(1+cut(2,1)):(end-cut(2,2)));
     end
  end
 end
 
   
   
   
 
%% WY UPDATE OLD (FASTER?) VERSION - SEEMS TO BE A BUG WITH GW ?
%   % Compute the changes in WY (the effect of the saved filter on nearby
%   % locations for all object types)
%   for mom1 = 1:opt.mom
%     for filt1_ind = 1:size(WY{type_hat}{mom1},3)
%       for obj_type = 1:opt.NSS
%         for filt2_ind = 1:size(WY{obj_type}{mom1},3)
%           WY{obj_type}{mom1}(inds{1},inds{2},filt2_ind) = ... 
%             WY{obj_type}{mom1}(inds{1},inds{2},filt2_ind) - ...
%             ( ...
%               GW{mom1}{filt1_ind,filt2_ind}(1+cut(1,1):end-cut(1,2),1+cut(2,1):end-cut(2,2)) * ... % The large interaction tensor
%               X(j, sum([0; n_regressors(1:mom1-1)])+filt1_ind) ... % The stored filter coefficient
%             );
%         end
%       end
%     end
%   end
%  
  
%% Update the changed xk values and delta log-likelihoods
  for obj_type=1:opt.NSS
    for mom1 = 1:opt.mom
      xk{obj_type}{mom1}(inds{1},inds{2},:) = ...
        mply(WY{obj_type}{mom1}(inds{1},inds{2},:), WnormInv{obj_type}{mom1}, 1);
      
      if strcmp(opt.learn_decomp, 'NMF')
        xk{obj_type}{mom1}(xk{obj_type}{mom1}<0) = 0; % Project onto positive only space
      end
%       % TOTHINK - Remove negative values
%       xk{obj_type}{mom1}(inds{1},inds{2},:) = ...
%         xk{obj_type}{mom1}(inds{1},inds{2},:) .* (xk{obj_type}{mom1}(inds{1},inds{2},:)>0);

%       dL_mom(inds{1},inds{2},obj_type,mom1) = ...
%         -sum(WY{obj_type}{mom1}(inds{1},inds{2},:) .* xk{obj_type}{mom1}(inds{1},inds{2},:),3);
    end
  end
  

    
  
%    figure(4); imagesc(WY(:,:,1)); colorbar; pause(0.05);  
 
  % Update the patch around the point found
%   Mask(max(row-3,1):min(row+3,end),max(col-3,1):min(col+3,end),type) = 0; % Make it impossible to put cells to close to eachother
%   Mask(max(row-1,1):min(row+1,end),max(col-1,1):min(col+1,end),:) = 0; % Make it impossible to put cells to close to eachother
    Mask(row_hat,col_hat,:) = 0; %Make it impossible to put a cell into the exact same location
  
if ~isempty(opt.spatial_push)
  [yinds, ycut] = mat_boundary(szWY(1:2),row_hat-opt.m:row_hat+opt.m,col_hat-opt.m:col_hat+opt.m);  
  [gridx,gridy] = meshgrid((-opt.m+ycut(1,1)):(opt.m-ycut(1,2)),(-opt.m+ycut(2,1)):(opt.m-ycut(2,2))); %make sure to cut the corresponding dimensions using ycut
  gridx = gridx'; gridy = gridy'; %meshgrid(1:n,1:m) creates mxn matrices, need to transpose
  
  grid_dist = sqrt(gridx.^2+gridy.^2);
  grid_dist = opt.spatial_push(grid_dist); % Specified distance based function
  Mask(yinds{1},yinds{2},:) = Mask(yinds{1},yinds{2},:).*repmat(grid_dist,[1,1,size(Mask,3)]); % Make it impossible to put cells to close to eachother
end
  
%  if opt.fig >3
% %   writeVideo(Video_dl, getframe(h_dl));
% %   writeVideo(Video_dl2, getframe(h_dl2));
% %   writeVideo(Video_dl3, getframe(h_dl3));
%   writeVideo(Video_comb, getframe(h_comb));
%  end  
%   disp([num2str(j) ' cells found, current type: ' num2str(type)]);
end

%   if opt.fig >3 
% %     close(Video_dl);
% %     close(Video_dl2);
% %     close(Video_dl3);
%     close(Video_comb);
%   end
% close(Video_yres);
end

