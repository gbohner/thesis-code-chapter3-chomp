function [ reconst, reconst_lowdim, Wfull ] = reconstruct_cell( opt, W, X, varargin )
%RECONSTRUCT_CELL Given a basis set W and corresponding coefficients X, reconstructs the mom-th moment of the cell 

%% Define inputs
p = inputParser();
p.addRequired('opt',@(x)isa(x,'chomp_options'));
p.addRequired('W',@isnumeric);
p.addRequired('X',@isnumeric);
p.addParameter('Wfull',{}); % If given, then no need to recompute the higher order regressors
p.addParameter('mom_max',opt.mom); % Return reconstructions up to this moment
p.addParameter('do_subset_combs',false); % IF true, use comb_inds to subset all combinations
p.addParameter('do_reconstruction',true);
p.parse(opt,W,X,varargin{:});

Wfull = p.Results.Wfull;
mom_max = p.Results.mom_max;
do_subset_combs = p.Results.do_subset_combs;
do_reconstruction = p.Results.do_reconstruction;

%% Compute the required new set of Ws 
if isempty(Wfull)
  Wfull = cell(opt.mom,1);
  for mom1 = 1:opt.mom
    [all_combs, mom_combs, comb_inds] = all_filter_combs(opt.KS, mom1, opt.diag_tensors);
    if do_subset_combs
      all_combs = all_combs(comb_inds,:); % Use only unique combinations - everything else is (super)symmetric - be careful to symmetrize during (re)constructions
    end
    if opt.diag_cumulants_extradiagonal && mom1>=2 % only do extra filters for non-mean 
       [extra_all_combs, ~, extra_comb_inds, ~] = all_filter_combs(opt.KS, mom1, 1);
       extra_all_combs = extra_all_combs(extra_comb_inds,:);
    else
      extra_all_combs=[];
    end
    
    if ~opt.diag_cumulants
      Wfull{mom1} = zeros((opt.m^2)^mom1, size(all_combs,1)+size(extra_all_combs,1)); 
    else
      Wfull{mom1} = zeros((opt.m^2), size(all_combs,1)); 
    end
    
    % For each row in all_combs compute the filter
    for filt1_ind = 1:size(all_combs,1)+size(extra_all_combs,1)
      if filt1_ind <= size(all_combs,1)
        filt1 = get_filter_comb(W, all_combs(filt1_ind,:), opt.diag_cumulants);
      else
        filt1_diag = get_filter_comb(W, extra_all_combs(filt1_ind-size(all_combs,1),:), 1); %have diag_cumulants=1 for this
        % This returned the diagonal of whatever tensor I want, augment it
        % appropriately;
        filt1 = zeros((opt.m.^2).^mom1,1);
        filt1(linspace(1,size(filt1,1),opt.m^2)) = filt1_diag(:); % fill in the nonzeros efficiently;
      end
      
      filt1 = filt1(:);
      if opt.diag_cumulants_offdiagonly && mom1>=2 % Blank out the diagonals of the high dimensional Ws
        filt1(linspace(1,numel(filt1),opt.m^2)) = 0;
      end
      
      % Compute the filter tensor
      Wfull{mom1}(:,filt1_ind) = filt1(:);
    end
      
    
  end
else
  assert(numel(Wfull)>=mom_max, 'CHOMP:reconstruct:wrongWfullGiven');
end


%% Compute the reconstructions

reconst = cell(mom_max,1);
reconst_lowdim = cell(mom_max,1);
curXstart = 0;
num_cells = size(X,1);


if do_reconstruction
  
  for mom1 = 1:mom_max
    % Compute reconstruction as an order mom1 tensor
    [all_combs, ~, comb_inds, comb_inds_rev] = all_filter_combs(opt.KS, mom1, opt.diag_tensors);
    Xcur = X(:,(curXstart+1):(curXstart+length(comb_inds))); % Extract relevant X weights for mom1
    if ~do_subset_combs % Extend X to be opt.KS^mom1 instead of the reduced representation
      Xcur = Xcur(:,comb_inds_rev);
    end
    curXstart = curXstart + length(comb_inds);
    
    if opt.diag_cumulants_extradiagonal && mom1>=2
      Xcur = [Xcur, X(:,(curXstart+1):(curXstart+opt.KS))]; % Add the corresponding parts from X
      curXstart = curXstart+opt.KS;
    end
    reconst{mom1} = Wfull{mom1}*Xcur';
    
    

    reconst_lowdim{mom1} = zeros(opt.m,opt.m,size(reconst{mom1},2));
    % Reshape into 2D images of cells
    for cell_num = 1:size(reconst_lowdim{mom1},3)
      if mom1==1 
          reconst_lowdim{mom1}(:,:,cell_num) = reshape(reconst{mom1}(:,cell_num),opt.m,opt.m);
      else
        % For higher moment just take the reconstruction on the diagonal to
        % show in the original (low-dim) space
        reconst_lowdim{mom1}(:,:,cell_num) = ...
          reshape(reconst{mom1}(linspace(1,size(reconst{mom1},1),opt.m^2),cell_num),opt.m,opt.m); % This is how to get tensor diagonal!
      end
    end
    
    % Reshape the reconstruction to a D*r dimensional tensor (D=2)
    if ~opt.diag_cumulants
      reconst{mom1} = reshape(reconst{mom1},[opt.m*ones(1,2*mom1), num_cells]);
    else
      reconst{mom1} = reshape(reconst{mom1},[opt.m, opt.m, num_cells]);
    end
  end

  
end

% -----------------------------------------------------------
% Symmetric higher order reconstruction tensor (unnecessary)
% -----------------------------------------------------------
% %% Compute the required new set of Ws 
% if isempty(Wfull)
%   Wfull = cell(opt.mom,1);
%   for mom1 = 1:opt.mom
%     [all_combs, ~, comb_inds] = all_filter_combs(opt.KS, mom1, opt.diag_tensors);
%     %all_combs = all_combs(comb_inds,:); % Use only unique combinations - everything else is (super)symmetric - be careful to symmetrize during (re)constructions
%     Wfull{mom1} = zeros((opt.m^2)^mom1, size(all_combs,1)); 
%     
%     % For each row in all_combs compute the filter
%     for filt1_ind = 1:size(all_combs,1)
%       % Compute the filter tensor
%       filt1 = get_filter_comb(W, all_combs(filt1_ind,:));
%       Wfull{mom1}(:,filt1_ind) = filt1(:);
%     end
%   end
% else
%   assert(numel(Wfull)>=mom_max, 'CHOMP:reconstruct:wrongWfullGiven');
% end
% 
% 
% %% Compute the reconstructions
% 
% reconst = cell(mom_max,1);
% reconst_lowdim = cell(mom_max,1);
% curXstart = 0;
% 
% for mom1 = 1:mom_max
%   % Compute reconstruction as an order mom1 tensor
%   [all_combs, ~, comb_inds, comb_inds_rev] = all_filter_combs(opt.KS, mom1, opt.diag_tensors);
%   Xcur = X(:,(curXstart+1):(curXstart+length(comb_inds)));
%   reconst{mom1} = Wfull{mom1}*Xcur(:,comb_inds_rev)';
%   curXstart = curXstart + length(comb_inds);
%   %reconst{mom1} = reshape(reconst{mom1},[opt.m^2*ones(1,mom1), size(reconst{mom1},2)]);
%   
%   reconst_lowdim{mom1} = zeros(opt.m,opt.m,size(reconst{mom1},2));
%   % Reshape into 2D images of cells
%   for cell_num = 1:size(reconst_lowdim{mom1},3)
%     if mom1==1 
%         reconst_lowdim{mom1}(:,:,cell_num) = reshape(reconst{mom1}(:,cell_num),opt.m,opt.m);
%     else
%       % For higher moment just take the reconstruction on the diagonal to
%       % show in the original (low-dim) space
%       reconst_lowdim{mom1}(:,:,cell_num) = ...
%         reshape(reconst{mom1}(linspace(1,size(reconst{mom1},1),opt.m^2),cell_num),opt.m,opt.m);
%     end
%   end
% end

end
