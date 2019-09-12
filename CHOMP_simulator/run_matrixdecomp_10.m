clear; 
timestamp = datestr(now,30);

newsamps = 1;
saving = 1;
toplot = 0;
showfig =0;
nsamps = 10000;

%Simulation params
runs = 200;
sparsity = 0.05;
num_ss = 2; %2
T = 1000; %200
I = [512];
num_obj = round(I*sparsity); %50
bsize = 11;
MOMs = [1,2,4];
random_basis = 1;
single_dist = 1;% 0, 0.5, 1;

if ~random_basis
  %non-random basis
  B(:,1) = [1,2,3,4,5,6,5,4,3,2,1]';
  B(:,2) = [1,2,1,2,3,4,3,2,1,2,1]';

  %Normalize basis funcs
  for i1 = 1:num_ss
    Bdims = cell(1,ndims(B)); [Bdims{1:end-1}] = deal(':');
    Bdims{end} = i1;
    B(Bdims{:}) = B(Bdims{:}) - mean(reshape(reshape(B(Bdims{:}),1,[]),size(B(Bdims{:})))); %zero mean
    B(Bdims{:}) = B(Bdims{:}) ./ norm(reshape(B(Bdims{:}),1,[])); %unit norm
  end
end

if saving
  save(['init_params' timestamp]);
end

%% Distribution function samples

if newsamps
%Generate lots of samples with each required mean variance and kurtosis,
%then allocate them to different "field of views"

% if ~random_basis
  means = [0 logspace(-2,1,4)]; %[0, 0.1, 0.5, 1, 10];%[0,1];
  vars = [logspace(-2,1,10)];
  kurs = [3,10,50]; %[3,5,10];
% else
%   means = [0 logspace(0,3,4)];
%   vars = [logspace(-1,3,5)];
%   kurs = [3,5,10];
% end

X = zeros(numel(vars), numel(kurs), nsamps);
for v1 = 1:numel(vars)
  for k1 = 1:numel(kurs)
    X(v1,k1,:) = get_dist([nsamps,1], [0,vars(v1),0,kurs(k1)]);
  end
end

end %newsamps

if saving
  save(['X_samps',timestamp],'X','means','vars','kurs','nsamps','-v7.3')
end

%% Add the samples to field of views then run the algorithms
results_found = zeros(numel(means), numel(vars), numel(kurs), numel(MOMs)+1);
results_count = zeros(numel(means), numel(vars), numel(kurs));
results_AUCs = zeros(runs,numel(MOMs)+1);
results_recalls = zeros(runs,numel(MOMs)+1);
results_runtimes = zeros(runs,numel(MOMs)+1);
results_runparams = zeros(runs,3); %mean, var, kurt if applicable

for r1 = 1:runs
  disp(r1)
  %Generate data
  L = randperm(490,num_obj)'+10; %True locations
  L = sort(L);
  Xsamps = zeros(num_obj,num_ss,T);
  Xdist = cell(num_obj,1);
  if single_dist
    Xdist_all = mat2cell(ceil(rand(3,1).*[numel(means),numel(vars),numel(kurs)]'),ones(3,1)); %Describes the distribution
    results_runparams(r1,:) = cell2mat(Xdist_all);
  end
  for o1 = 1:num_obj
    %For each object draw a distribution and use samples from that dist
    if single_dist==1
      Xdist{o1} = Xdist_all;
    elseif single_dist==0
      Xdist{o1} = mat2cell(ceil(rand(3,1).*[numel(means),numel(vars),numel(kurs)]'),ones(3,1)); %Describes the distribution
    elseif single_dist==0.5 % sample close to the true distribution
      Xdist_allmat = cell2mat(Xdist_all);
      Xdist_allmat = Xdist_allmat + round(randn(3,1).*[0.95;1.7;1.38]); %Set variances to have some fair statement about order of magnitude
      Xdist_allmat(Xdist_allmat<1)=1;
      Xdist_allmat(1) = min(Xdist_allmat(1),numel(means));
      Xdist_allmat(2) = min(Xdist_allmat(2),numel(vars));
      Xdist_allmat(3) = min(Xdist_allmat(3),numel(kurs));
      Xdist{o1} = mat2cell(Xdist_allmat,ones(3,1)); %Describes the distribution
    end
    cursamps = ceil(rand(T*num_ss,1)*nsamps);
    Xsamps(o1,:,:) = reshape(X(Xdist{o1}{2:end},cursamps),num_ss,T); %Get signal coefficients with give variance and kurtosis
    %Enforce correct sample mean
    Xsamps(o1,:,:) = Xsamps(o1,:,:) - reshape(mean(reshape(Xsamps(o1,:,:),1,[]))*ones(1,numel(Xsamps(o1,:,:))),size(Xsamps(o1,:,:))) + means(Xdist{o1}{1});
    results_count(Xdist{o1}{:}) = results_count(Xdist{o1}{:})+1; %Count the distributions we use
  end
  if random_basis
    B = randn(bsize, num_ss);
    %B = B+flipud(B); %symmetric basis funcs
      %Normalize basis funcs
    for i1 = 1:num_ss
      Bdims = cell(1,ndims(B)); [Bdims{1:end-1}] = deal(':');
      Bdims{end} = i1;
      B(Bdims{:}) = B(Bdims{:}) - mean(reshape(reshape(B(Bdims{:}),1,[]),size(B(Bdims{:})))); %zero mean
      B(Bdims{:}) = B(Bdims{:}) ./ norm(reshape(B(Bdims{:}),1,[])); %unit norm
    end
  end
    
  y = create_data(I,T,L,Xsamps,B,1); % Generated data
  true{r1} = struct('y_mean',mean(y,ndims(y)),'y_var',var(y,0,ndims(y)),'L',L,'Xsamps',Xsamps,'Xdist',{Xdist});
   
  
 
  
  
  mom_perf = zeros(1,numel(MOMs)+1);
  %Run the algorithm for different moments
  for mom1 = 1:numel(MOMs)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INSTEAD OF THIS SOLVER
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%
    
    tic;
    M = get_moments( y, B, MOMs(mom1), 'diagonal', 1); %moments by subspace by data points
    M = reshape(M,MOMs(mom1),num_ss,[]); %easier to use format
    O = matching_pursuit( M(1:MOMs(mom1),:,:), B, I, num_obj, showfig);
    results_runtimes(r1,mom1) = toc; %sec
    
    
    % Evaluate some metric on how good it was, say number of locations
    % found:
    Lest = [O.loc]';
    
    
    results_AUCs(r1,mom1) = get_AUC(L, Lest);
    results_recalls(r1,mom1) = get_AUrecall(L, Lest, I);
    [idx, dl] = knnsearch(Lest,L);
    %dl = abs(L - Lest(idx)); %distances between true locations and closest estimate
    found = zeros(size(L));
    found(dl<=0) = 1; %Arbitrary thresholding of "finding" correct location, could be dl==0 for perfect
    %For each correctly find object, add 1 to the respective squares
    for o1 = 1:num_obj
      if found(o1)
        results_found(Xdist{o1}{:}, mom1) = results_found(Xdist{o1}{:}, mom1) + 1;
      end
    end
    
    mom_perf(mom1) = sum(found)/numel(found);
%     % For diagnostics
%     if (mom1==1) && (sum(found)>2), 
%       %matching_pursuit( M(1:mom1,:,:), B, I, num_obj,1); title(mom1); 
%       pause; 
%     end
    est{r1,mom1} = struct('O',O,'L',L,'Lest',Lest,'found',found);
  end
  
  
  tic;
  [Lest, nx1] = matrix_decomp_localise(y, B, I, num_obj);
  results_runtimes(r1,numel(MOMs)+1) = toc;
  
  results_AUCs(r1,numel(MOMs)+1) = get_AUC(L, Lest);
  results_recalls(r1,numel(MOMs)+1) = get_AUrecall(L, Lest, I);
  [idx, dl] = knnsearch(Lest,L);
  %dl = abs(L - Lest(idx)); %distances between true locations and closest estimate
  found = zeros(size(L));
  found(dl<=0) = 1;
  mom_perf(numel(MOMs)+1) = sum(found)/numel(found);
  
  for o1 = 1:num_obj
    if found(o1)
      results_found(Xdist{o1}{:}, numel(MOMs)+1) = results_found(Xdist{o1}{:}, numel(MOMs)+1) + 1;
    end
  end
  
  est{r1,numel(MOMs)+1} = struct('O',nx1,'L',L,'Lest',Lest,'found',found);
  
  
  disp([mom_perf; results_AUCs(r1,:); results_runtimes(r1,:)])
  if saving && (mod(r1,1)==0)
    if saving==2
      save(['figure_5_results' timestamp],'true','est','results*','-v7.3');
    else
      save(['figure_5_results' timestamp],'results*','est','-v7.3');
    end
  end
end