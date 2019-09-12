% One dimensional simulations, 512 locations, 5 objects, T = 300

%Simulation params
num_obj = 5; %50
num_ss = 2; %2
T = 200; %200
I = [512];
bsize = 11;

%non-random basis
B(:,1) = [1,2,3,4,5,6,5,4,3,2,1]';
B(:,2) = [1,2,1,2,3,4,3,2,1,2,1]';

%Normalize basis funcs
for i1 = 1:num_ss
  Bdims = cell(1,ndims(B)); [Bdims{1:end-1}] = deal(':');
  Bdims{end} = i1;
  B(Bdims{:}) = B(Bdims{:}) ./ norm(reshape(B(Bdims{:}),1,[]));
end

%% Create a set of signal distribution (noise is always unit variance white additive)

%Step
dists{1} = @(T,SNR,num_obj,num_ss)ones(num_obj,num_ss,T)*SNR;
%Normal
dists{2} = @(T,SNR,num_obj,num_ss)randn(num_obj,num_ss,T)*SNR;
%Laplace
dists{3} = @(T,SNR,num_obj,num_ss)reshape(laprnd(num_obj*num_ss,T,0,SNR),num_obj, num_ss, T);
% %Uniform
% dists{4} = @(T,SNR,num_obj,num_ss)rand(num_obj,num_ss,T)*SNR*2;

% Skewed exponential distritubion
dists{4} = @(T,SNR,num_obj,num_ss)exprnd(SNR,num_obj,num_ss,T)-SNR; % high skewness


%% Create the other parameters we are iterating over
MOMs = 1:4;
SNRs = [0.01,0.1,0.25,0.5,1,5,25,100];
runs = 100;

%% Run simulations and save results

true = cell(numel(dists),numel(SNRs),runs); % distributions x SNR conditions x runs
est = cell(numel(dists),numel(SNRs),runs,numel(MOMs)); % distributions x  SNR conditions x runs x moments used
all_perf = zeros(numel(dists),numel(SNRs),runs,numel(MOMs));


for snr1 = 1:numel(SNRs)
  for d1 = 1:4 %Distributions
    for r1 = 1:runs
      disp([snr1,d1,r1])
      %Generate data
      L = floor(rand(num_obj,numel(I)) * 490 + 10); %True locations
      X = dists{d1}(T,SNRs(snr1),num_obj,num_ss); %True signal coefficients
      y = create_data(I,T,L,X,B,1); % Generated data
      true{d1,snr1,r1} = struct('y_mean',mean(y,ndims(y)),'y_var',var(y,0,ndims(y)),'L',L,'X',X);
      
      M = get_moments( y, B, max(MOMs), 'diagonal', 1); %moments by subspace by data points
      M = reshape(M,max(MOMs),num_ss,[]); %easier to use format
      
      %Run the algorithm for different moments
      for mom1 = 1:numel(MOMs)
        O = matching_pursuit( M(1:mom1,:,:), B, I, num_obj);
        
        % Evaluate some metric on how good it was, say number of locations
        % found:
        Lest = [O.loc]';
        idx = knnsearch(L, Lest);
        dl = abs(L(idx) - Lest); %distances
        found = zeros(size(L));
        found(dl<=0) = 1; %Arbitrary thresholding of "finding" correct location, could be dl==0 for perfect
        perf = sum(found)/numel(found); %performance
        
        all_perf(d1,snr1,r1,mom1) = perf;
        est{d1,snr1,r1,mom1} = struct('O',O,'L',L,'Lest',Lest,'perf',perf);
      end
    end
  end
end
  
  

save('figure_4_results','true','est','all_perf','-v7.3');



%% Loading and using results
% load('figure_4_results')
% for SNR = 1:numel(SNRs)
%   for d1 = 1:4 %Distributions
%     for r1 = 1:runs
%       for mom1 = 1:numel(MOMs)
%          % Evaluate some metric on how good it was, say number of locations
%         % found:
%         O = est{d1,SNR,r1,mom1}.O;
%         L = est{d1,SNR,r1,mom1}.L;
%         Lest = [O.loc]';
%         idx = knnsearch(L, Lest);
%         dl = abs(L(idx) - Lest); %distances
%         found = zeros(size(L));
%         found(dl==0) = 1; %Arbitrary thresholding of "finding" correct location, could be dl==0 for perfect
%         perf = sum(found)/numel(found); %performance
%         
%         all_perf(d1,SNR,r1,mom1) = perf;
%       end
%     end
%   end
% end

%% Visualize results
dist_names = {'Step','Normal','Laplace','Exponential'};
colors = ['r','g','b','m'];
%linetype = {'-x','-.o','--.','-.*',':d',':o', ':*',':.'};
cum_plot = [1,2,3,4];
figure; %hold on;
for snr1 = 1:8
  subplot(2,4,snr1); hold on
  for d1 = 1:4
    h_plot(d1,snr1) = plot(1:numel(cum_plot), squeeze(mean(all_perf(d1,snr1,:,cum_plot),3)), [colors(d1)]);
    xlim([1,numel(cum_plot)])
    ylim([0,1])
    set(gca,'XTick',1:numel(cum_plot))
    set(gca,'XTickLabel',cum_plot)
    title(sprintf('SNR: %0.2f',SNRs(snr1)))
  end
end
legend(h_plot(:,1), dist_names,'Location','NorthEast');
xlabel('Cumulants used')
ylabel('True positive rate')
%Colors only bugged in R2015b matlab otherwise should be fine...


