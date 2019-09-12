function [opt, model] = load_best_fits(dataset_id, env_home)
%LOAD_BEST_OPTS Summary of this function goes here
%   Detailed explanation goes here

if ~exist('env_home','var')
  env_home = 0;
end

% preProcessing params
prior = 'expertPrior';%prior = 'noPrior';
lik = 'unampLik'; %lik = 'linLik';
gitsha = 'gitsha_2bd0d72_evalgit_db4ade8';
trainType = '_rPC_1_origPMgain_useNans';
targetCoverage = '_targetCoverage_10';
gridType = '_grid_30_7';

if strcmp(dataset_id, '00.00')
  % Load preprocessed model fit
  dataset_name = 'neurofinder.00.00';
  chomp_iter = '2';
  chomp_timestamp = '20190527T202351';
elseif strcmp(dataset_id, '01.00')
  chomp_timestamp ='20190529T060454'; % 2_proper run (var only)
  dataset_name = 'neurofinder.01.00';
  chomp_iter = '4';
elseif strcmp(dataset_id, '02.00')
  dataset_name = 'neurofinder.02.00';
%   chomp_timestamp ='20190529T060454'; % 2_proper run (var only) 
%   chomp_iter = '7'; 
  chomp_timestamp ='20190607T192326';  % noimpute run (covar only)
  gitsha = [gitsha '_noimpute']; 
  chomp_iter = '2';
elseif strcmp(dataset_id, '03.00')
  dataset_name = 'neurofinder.03.00'; 
  chomp_timestamp = '20190527T225215'; % 0.84! best yet. This is supervised atm, but seems to do fine without supervision too!
  chomp_iter = '2';
elseif strcmp(dataset_id, '04.00')
  dataset_name = 'neurofinder.04.00';
  chomp_timestamp = '20190517T221446'; % 0.28
  gitsha = 'gitsha_2bd0d720de0995be6b0f1795304839f9877cb6c3';
  chomp_iter = '2';
end
  
  

stamp = ['_preproc2P_' prior '_' lik '_' gitsha trainType targetCoverage gridType];
  
  
if env_home

  cur_model = load(['/mnt/gatsby/nfs/data/gergo/Neurofinder_update/' ...
    dataset_name '/preproc2P/CHOMP/output/' dataset_name ...
    stamp '_' ...
    chomp_timestamp '_iter_' chomp_iter '.mat']);

  model = cur_model.model;
  model.opt.root_folder = '/mnt/gatsby';
  opt = model.opt;
  
else % on office computer
  cur_model = load(['/nfs/data/gergo/Neurofinder_update/' ...
  dataset_name '/preproc2P/CHOMP/output/' dataset_name ...
  stamp '_' ...
  chomp_timestamp '_iter_' chomp_iter '.mat']);

  model = cur_model.model;
  opt = model.opt;
end

end

