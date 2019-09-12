

get_area_inds = @(cur_area){cur_area(1):cur_area(3), cur_area(2):cur_area(4)};

% Neurofinder 00
data_path = '/nfs/data/gergo/Neurofinder_update/neurofinder.00.00/preproc2P/images_expertPrior_unampLik_gitsha_2bd0d72_evalgit_db4ade8_rPC_1_origPMgain_useNans_targetCoverage_10_grid_30_7/image00001.tif';

% Zoom 1 (area_51, middle)
area_51 = [285,70,400,185];
createNeurofinderVideo(data_path, '.tif', 1:2000, [], get_area_inds(area_51))

% Zoom 2 (area_52, top right)
area_52 = [5,391,121,506];
createNeurofinderVideo(data_path, '.tif', 1:2000, [], get_area_inds(area_52), [], 0)

% Without zoom
createNeurofinderVideo(data_path, '.tif', 1:2000, [], [], [], 0)


data_path = '/nfs/data/gergo/Neurofinder_update/neurofinder.01.00/preproc2P/images_expertPrior_unampLik_gitsha_2bd0d72_evalgit_db4ade8_rPC_1_origPMgain_useNans_targetCoverage_10_grid_30_7/image00001.tif';

createNeurofinderVideo(data_path, '.tif', 1:2000, [], [], [], 0)

data_path = '/nfs/data/gergo/Neurofinder_update/neurofinder.02.00/preproc2P/images_expertPrior_unampLik_gitsha_2bd0d72_evalgit_db4ade8_rPC_1_origPMgain_useNans_targetCoverage_10_grid_30_7/image00001.tif';

createNeurofinderVideo(data_path, '.tif', 1:2000, [], [], [], 0)

