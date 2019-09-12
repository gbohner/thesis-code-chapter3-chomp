function [O, scores_init] = matching_pursuit( M, B, I, num_obj, fig)
%MATCHING_PURSUIT matching pursuit step for diagonal moment tensors and 2d
%data

if nargin<5, fig=0; end;

% Get derived params
m = size(B,1);
num_ss = size(B,ndims(B));
mom = size(M,1); %needs to be changed later
M = reshape(M,mom,num_ss,[]); %easier to use format


%Compute shift tensor
if exist('precomputed/cur.mat','file')
    load('precomputed/cur.mat','GPT')
else
  if ndims(B)==3
    GPT = precompute_shift_tensors(struct('m',m,'mom',mom));    
  elseif ndims(B)==2
    GPT = precompute_shift_tensors_1d(struct('m',m,'mom',mom));
  end
end

% Compute filter shifts
W = reshape(B,[prod(size(B))/size(B,ndims(B)), size(B,ndims(B))]);
Worig = W;

for filt1 = 1:size(W,2)
  for filt2 = 1:size(W,2)
    Wcur1 = Worig(:,filt1);
    Wcur2 = Worig(:,filt2);
    for mom1 = 1:mom
      if mom1>1, Wcur1 = mply(Wcur1, Worig(:,filt1)',0); end
      if mom1>1, Wcur2 = mply(Wcur2, Worig(:,filt2)',0); end

%       %TODO flip all dimensions of the second filter, such that convolution
%       %gives you nd correlation instead
%       Wcur2r = Wcur2;
%       Wcur2r = flipdim_all(Wcur2r);
      Wcur1 = Wcur1./norm(Wcur1(:)+1e-6); % make sure it has norm of 1.
      Wcur2 = Wcur2./norm(Wcur2(:)+1e-6); % make sure it has norm of 1.
      Wcur1c = Wcur1(:);
      Wcur2c = Wcur2(:);

      if ndims(B)==3
        GW{filt1,filt2,mom1} = zeros(2*m-1, 2*m-1);
      elseif ndims(B)==2
        GW{filt1,filt2,mom1} = zeros(2*m-1, 1);
      end
      for s1 = 1:(2*m-1)
        if ndims(B)==3
          for s2 = 1:(2*m-1)
            GW{filt1,filt2,mom1}(s1,s2) = Wcur2c(GPT{s1,s2,mom1}(:,2))'* Wcur1c(GPT{s1,s2,mom1}(:,1)); %compute the shifted effect in original space via the shift tensors GPT. Because the Worigs were computed to correspond to the best inverse of the Ws
          end
        elseif ndims(B)==2
          GW{filt1,filt2,mom1}(s1,1) = Wcur2c(GPT{s1,1,mom1}(:,2))'* Wcur1c(GPT{s1,1,mom1}(:,1)); %compute the shifted effect in original space via the shift tensors GPT. Because the Worigs were computed to correspond to the best inverse of the Ws
        end
      end
    end
  end
end

WnormInv = zeros(size(W,2),size(W,2),mom); % Inverse Interaction between basis functions

%WnormInv should be in feature space (when doing random projection version)
for filt1 = 1:size(W,2)
  for filt2 = 1:size(W,2)
    Wcur1 = W(:,filt1);
    Wcur2 = W(:,filt2);
    for mom1 = 1:mom
      if mom1>1, Wcur1 = mply(Wcur1, W(:,filt1)',0); end
      if mom1>1, Wcur2 = mply(Wcur2, W(:,filt2)',0); end
      Wcur1 = Wcur1./norm(Wcur1(:)+1e-6); % make sure it has norm of 1.
      Wcur2 = Wcur2./norm(Wcur2(:)+1e-6); % make sure it has norm of 1.

      WnormInv(filt1,filt2,mom1) = Wcur1(:)'*Wcur2(:); 
    end
  end
end

%Invert Wnorm
for mom1 = 1:mom
  WnormInv(:,:,mom1) = inv(WnormInv(:,:,mom1)+1e-6*eye(size(WnormInv,1),size(WnormInv,2))); % Regularized
end

% Run matching pursuit
O = struct(); %solution set
Ucur = zeros(mom*num_ss,1);
%num_obj = 50;
for iter = 1:num_obj

    %Mahalonobis distance from noise distribution
%     M = reshape(M,size(M,1)*size(M,2),[]);
%     scores = mahal(M',M(:,sum(abs(M)<repmat(quantile(abs(M),0.8,2),[1,size(M,2)]),1)==size(M,1))')'; %how far samples are from the "noise dist"
%     M = reshape(M,mom,num_ss,[]);
    
    %Use the cost function derived
    for mom1 = 1:mom
      scores_all(mom1,:) = diag(squeeze(M(mom1,:,:))' * WnormInv(:,:,mom1)' * squeeze(M(mom1,:,:)));
    end
    
    %Use the optimal weightings based on cost func
    for mom1 = 1:mom
      to_mply_with(mom1,1) = 1./(m^mom1 - num_ss^mom1);
    end
    scores = to_mply_with'*scores_all;
    
    
    if iter==1
      scores_init = scores;
    end
    
%     %Let's not used 3d moment
%     if mom>=3
%       scores_all(3,:) = [];
%     end
    
    %Get the Mahalonobis distance in the reconstruction space
    %scores = mahal(scores_all',scores_all(:,sum(abs(scores_all)<repmat(quantile(abs(scores_all),0.95,2),[1,size(scores_all,2)]),1)==size(scores_all,1))')'; %how far samples are from the "noise dist"
    %scores = mahal(scores_all',scores_all')'; % This was used for paper
    %version at 30 April
    %scores = sum(scores_all,1);
    %Diagnostic for which moments contribute to the score how much
    %[~,ind] = max(scores); mu = mean(scores_all'); sig=cov(scores_all'); ((scores_all(:,ind) - mu')'/sig).*(scores_all(:,ind) - mu')'
    
    
    % Visualize scores
    if fig
        figure(346); subplot(1,5,mod(iter,5)+5*(rem(iter,5)==0)); 
        %Mz = zscore(reshape(M,mom*num_ss,[])')';
        plot(squeeze(M(1:mom,1,:))'); %legend('1','2','3','4')
        figure(347); 
        subplot(1,5,mod(iter,5)+5*(rem(iter,5)==0));
        if ndims(B)==3
          imagesc(reshape(scores,I));
          colorbar;
        elseif ndims(B)==2
          plot(scores);
        end
        pause(0.01)
    end
    
    %Alternative "skewness" model (we really need something relatively
    %stable here?
    %scores_sorted = sort(scores,'descend');
    %if mean(scores_sorted(1:num_obj)) - median(scores_sorted) > median(scores_sorted) - mean(scores_sorted(end-num_obj+1:end))
%     if iter==1
%       skewdir = 2*(skewness(scores)>=0) - 1;
%       %skewness(scores)
%     end
    skewdir = 1;

    if skewdir==1
        [~, inds] = sort(abs(scores),'descend');
    else
        [~, inds] = sort(abs(scores),'ascend');
    end
    
    %(1:5)
    %scores(inds(1:5))
    
    locs = 1:prod(I);
    locs_sorted = locs(inds);
    L_estimate = cell(numel(I),1);
    [L_estimate{:}] = ind2sub(I,locs_sorted(1:num_obj));

    L_estimate = cell2mat(L_estimate);
    
    lstar = L_estimate(:,1);
    mstar = M(:,:,inds(1));
    O(iter).loc = lstar;
    O(iter).moments = mstar;
    
    
    
    
    %Get change in M
    for m1 = 1:mom
        sol = squeeze(M(m1,:,inds(1)))*WnormInv(:,:,m1); %best reconst coeffs cause of correlated unnormalized filters
        for k1 = 1:num_ss
            for s1 = 1:2*m-1
              if ndims(B)==3
                for s2 = 1:2*m-1
                    %Updating M using GW (backproj then reproj with filters
                    %k2 and k1
                    try
                      lcur = sub2ind(I,lstar(1)+s1-m,lstar(2)+s2-m);
                    catch
                      continue;
                    end
                    
                    
                    for k2 = 1:num_ss
                        M(m1,k1,lcur) = M(m1,k1,lcur) - GW{k2,k1,m1}(s1,s2)*sol(k2);
                        %M(m1,k1,lcur) = M(m1,k1,lcur) - GW{k2,k1,m1}(s1,s2)*M(m1,k2,inds(1)); %Without WnormInv
                    end
                end
              elseif ndims(B)==2
                try
                  lcur = lstar(1)+s1-m;
                  assert(lcur>0);
                  assert(lcur<=I);
                catch
                  continue;
                end
                for k2 = 1:num_ss
                    M(m1,k1,lcur) = M(m1,k1,lcur) - GW{k2,k1,m1}(s1,1)*sol(k2);
                    %M(m1,k1,lcur) = M(m1,k1,lcur) - GW{k2,k1,m1}(s1,s2)*M(m1,k2,inds(1)); %Without WnormInv
                end
              end
                
            end
        end
    end


end

