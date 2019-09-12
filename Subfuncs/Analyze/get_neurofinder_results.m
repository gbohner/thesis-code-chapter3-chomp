function out = get_neurofinder_results( opt, ROIs, is_preproc2P, varargin )
%GET_NEUROFINDER_RESULTS Summary of this function goes here
%   Detailed explanation goes here
% Make sure that the "neurofinder" script is on Matlab's PATH
% Returns out = [num_cell, recall, precision, combined]
PATH = getenv('PATH');

% % Version for local laptop
% if isempty(strfind(PATH, '/Users/gergobohner/anaconda2/bin'))
%   setenv('PATH', [PATH ':/Users/gergobohner/anaconda2/bin']);
% end

% Version for office computer
if isempty(strfind(PATH, '/nfs/nhome/live/gbohner/anaconda2/bin'))
  setenv('PATH', [PATH ':/nfs/nhome/live/gbohner/anaconda2/bin']);
end

% Version for mac
if isempty(strfind(PATH, '/usr/local/bin'))
  setenv('PATH', [PATH ':/usr/local/bin']);
end

if nargin < 3
    is_preproc2P = false;
end

if is_preproc2P
    orig_path = fileparts(fileparts(fileparts(fileparts(get_path(opt)))));
else
    orig_path = fileparts(fileparts(fileparts(get_path(opt))));
end
    
%figure;

out = [];

if nargin>3
  num_cells_iters = varargin{1};
else
  num_cells_iters = [10, 30, 50:50:numel(ROIs), numel(ROIs)];
end
  

for num_cells = num_cells_iters
  
    eval_command = ['neurofinder evaluate ' orig_path '/regions/regions.json'...
      ' ' ROI_to_json(opt, ROIs, num_cells)];
    [status,cmdout] = unix(eval_command,'-echo');

    recall = str2num(cmdout((strfind(cmdout, '"recall"') + 10):(strfind(cmdout, '"combined"') - 3)));
    combined = str2num(cmdout((strfind(cmdout, '"combined"') + 12):(strfind(cmdout, '"precision"') - 3)));
    precision = str2num(cmdout((strfind(cmdout, '"precision"') + 13):(strfind(cmdout, '"inclusion"') - 3)));
    
    out = [out; num_cells, recall, combined, precision];
    

    %scatter(num_cells, 2*recall*precision/(recall+precision)); hold on;
end

end

