function createNeurofinderVideo(data_path, src_string, use_frames, setcolormap, inds, outpath, medianfilter)
%CREATENEUROFINDERVIDEO Summary of this function goes here
%   Detailed explanation goes here


    filepath = [fileparts(data_path) filesep];
    allfiles = dir([filepath '*' src_string '*']);
    
    if nargin<7
      medianfilter = 0;
    end
    
    if nargin<6, outpath = []; end
    if isempty(outpath)
      nf_id = strsplit(filepath, '/'); % for office computer
      nf_id = nf_id{6};
      outpath = ['/nfs/nhome/live/gbohner/public_html/Thesis/savedVideos/'...
        nf_id, '_' datestr(now,30)];
    end
        
    
    if nargin < 3, use_frames=[]; end
    if isempty(use_frames)
      T = size(allfiles,1);
    else
      T = numel(use_frames);
      allfiles = allfiles(use_frames);
    end
    
    if nargin<4, setcolormap = []; end
    if isempty(setcolormap)  
      %setcolormap = gray(256);
      my_colormaps = load('Examples/matlab_colormaps.mat');
      setcolormap = my_colormaps.felfire;
    end
    
    
    
    
    im_cur = single(imread([filepath allfiles(1).name]));
    
    if nargin<5, inds = []; end
    if isempty(inds)
      inds = {1:size(im_cur,1), 1:size(im_cur,2)};
    end
    sz = [numel(inds{1}), numel(inds{2})];
    
    Y = zeros([sz(1), sz(2), T]); % Image stack
    
    Y(:, :,1) = im_cur(inds{1},inds{2});
    for i2 = 2:T
      im_cur = single(imread([filepath allfiles(i2).name]));
      Y(:, :,i2) = im_cur(inds{1},inds{2});
    end
    
    if medianfilter
      Y = medfilt3(Y, [1,1,5]); 
    end
    
    
    v = VideoWriter(outpath,'Motion JPEG AVI');
    v.Quality = 70;
    v.FrameRate = 30;
    
    % Add color to Y
    Y = discretize(Y, size(setcolormap,1));
    Ycolor = repmat(Y,[1,1,1,3]);
    for i1 = 1:3
      Ycolor(:,:,:,i1) = reshape(setcolormap(Y(:),i1),size(Y));
    end
    Ycolor = permute(Ycolor,[1,2,4,3]); % change color to be 3rd dim
    
    open(v);
    writeVideo(v, Ycolor);
    close(v);
end

