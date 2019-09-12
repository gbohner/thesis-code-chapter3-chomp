fname = ['/nfs/data/gergo/Neurofinder_update/final_submission_file_combined_best_', datestr(now, 30), '.json'];

results = [];

% For each submission file, pick the "sources_" file we want to submit


timestamp_00 = '20190527T202351';

% Dataset 00.00.test
cur_regions=loadjson(['/nfs/data/gergo/Neurofinder_update/neurofinder.00.00.test/preproc2P/CHOMP/results/' ...
  'sources_' timestamp_00 '.json']);
cur_regions_struct = struct('coordinates',[]);
for i1 = 1:length(cur_regions)
  cur_regions_struct(i1).coordinates = cur_regions{i1}.coordinates;
end
results = [results, struct('dataset', '00.00.test', 'regions', cur_regions_struct)];

% Dataset 00.01.test
cur_regions=loadjson(['/nfs/data/gergo/Neurofinder_update/neurofinder.00.01.test/preproc2P/CHOMP/results/' ...
  'sources_' timestamp_00 '.json']);
cur_regions_struct = struct('coordinates',[]);
for i1 = 1:length(cur_regions)
  cur_regions_struct(i1).coordinates = cur_regions{i1}.coordinates;
end
results = [results, struct('dataset', '00.01.test', 'regions', cur_regions_struct)];

%-------------------------------------------------------------


% Dataset 01.00.test
cur_regions = loadjson('/nfs/data/gergo/Neurofinder_update/neurofinder.01.00.test/CHOMP/results/sources_20170727T144747.json');
cur_regions_struct = struct('coordinates',[]);
for i1 = 1:length(cur_regions)
  cur_regions_struct(i1).coordinates = cur_regions{i1}.coordinates;
end
results = [results, struct('dataset', '01.00.test', 'regions', cur_regions_struct)];


% Dataset 01.01.test
cur_regions = loadjson('/nfs/data/gergo/Neurofinder_update/neurofinder.01.01.test/CHOMP/results/sources_20170727T144747.json');
cur_regions_struct = struct('coordinates',[]);
for i1 = 1:length(cur_regions)
  cur_regions_struct(i1).coordinates = cur_regions{i1}.coordinates;
end
results = [results, struct('dataset', '01.01.test', 'regions', cur_regions_struct)];



% Dataset 02.00.test
cur_regions = loadjson('/nfs/data/gergo/Neurofinder_update/neurofinder.02.00.test/CHOMP/results/sources_20170816T101642.json');
cur_regions_struct = struct('coordinates',[]);
for i1 = 1:length(cur_regions)
  cur_regions_struct(i1).coordinates = cur_regions{i1}.coordinates;
end
results = [results, struct('dataset', '02.00.test', 'regions', cur_regions_struct)];

%-------------------------------------------------------------
timestamp_02 = '20190516T153006';



% Dataset 02.01.test
cur_regions=loadjson(['/nfs/data/gergo/Neurofinder_update/neurofinder.02.01.test/preproc2P/CHOMP/results/' ...
  'sources_' timestamp_02 '.json']);
cur_regions_struct = struct('coordinates',[]);
for i1 = 1:length(cur_regions)
  if ~isempty(cur_regions{i1}.coordinates)
    cur_regions_struct(i1).coordinates = cur_regions{i1}.coordinates;
  end
end
results = [results, struct('dataset', '02.01.test', 'regions', cur_regions_struct)];

% Dataset 03.00.test
timestamp_03 = '20190527T225215'; % 0.84! best yet. This is supervised atm, but seems to do fine without supervision too!

% Dataset 03.00.test
cur_regions=loadjson(['/nfs/data/gergo/Neurofinder_update/neurofinder.03.00.test/preproc2P/CHOMP/results/' ...
  'sources_' timestamp_03 '.json']);
cur_regions_struct = struct('coordinates',[]);
for i1 = 1:length(cur_regions)
  cur_regions_struct(i1).coordinates = cur_regions{i1}.coordinates;
end
results = [results, struct('dataset', '03.00.test', 'regions', cur_regions_struct)];




% Dataset 04.00.test
timestamp_04_00 = '20190517T221446'; % 0.28

cur_regions=loadjson(['/nfs/data/gergo/Neurofinder_update/neurofinder.04.00.test/preproc2P/CHOMP/results/' ...
  'sources_' timestamp_04_00 '.json']);
cur_regions_struct = struct('coordinates',[]);
for i1 = 1:length(cur_regions)
  if ~isempty(cur_regions{i1}.coordinates)
    cur_regions_struct(i1).coordinates = cur_regions{i1}.coordinates;
  end
end
results = [results, struct('dataset', '04.00.test', 'regions', cur_regions_struct)];


% Dataset 04.01.test
cur_regions = loadjson('/nfs/data/gergo/Neurofinder_update/neurofinder.04.01.test/CHOMP/results/sources_20170727T153355.json');
cur_regions_struct = struct('coordinates',[]);
for i1 = 1:length(cur_regions)
  cur_regions_struct(i1).coordinates = cur_regions{i1}.coordinates;
end
results = [results, struct('dataset', '04.01.test', 'regions', cur_regions_struct)];


savejson('', results, fname);

[~, fname, fext] = fileparts(fname);

% Auto save to CHOMP folder too
fname = ['/nfs/nhome/live/gbohner/Dropbox_u435d_unsynced_20190502/Gatsby/Research/CHOMP/' fname, fext];
savejson('', results, fname);



