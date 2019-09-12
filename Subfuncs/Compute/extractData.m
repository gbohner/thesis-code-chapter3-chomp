function opt = extractData( opt )
%EXTRACTDATAFROMTIF Extracts the relavant information from an input tif
%stack 
% - give input tif file name
% - output path for large .mat file
% - other arguments:
%   - options struct
%   -  ...

  opt=opt; %avoids matlab error of "output not assigned..."

  %Set the path for the intermediate preprocessed "input" file
  intermediate_path = get_path(opt);
  
%Check if we want to just load an already preprocessed file
if exist(intermediate_path, 'file')
  %If it already exists, check if the preprocessing options are the same, and just load the files
  inp = load(intermediate_path, 'inp');
  inp = inp.inp;
  
  if ~struct_contain(opt, inp.opt, {'spatial_scale', 'time_scale', 'whiten', 'smooth_filter_mean' , 'smooth_filter_var'})
    error('CHOMP:preprocess:outdatedoptions', ...
      ['The new option struct has different preprocessing options than the', ...
       'intermediate file you want to use it with, consider removing manual timestamp' ...
       'from your input options file to create a new preprocessed file']);
  else
    if ~struct_contain(opt, inp.opt) %some settings has changed, give warning, and let user create a new copy of the file with new timestamp
      if opt.verbose>1
        warning('CHOMP:preprocess:outdatedoptions', 'The new option struct has slightly different options than the intermediate file you want to use it with.');
      end
      inp.opt = struct_merge(inp.opt, opt);
      intermediate_path = get_path(inp.opt);
      save(intermediate_path, 'inp', '-v7.3'); %overwrite the old input file
    end
  end
     
  
else %Handle the data loading-preprocessing-saving

  %Set the input data path
  data_path = [opt.root_folder opt.data_path];
  
  if ~exist(data_path,'file')
      error('CHOMP:preprocess:noinputdata', ...
      ['There is no file at the given input destination to read']);
  end
  
  %Read the data into a preprocessed and a raw stack, as well as store the
  %original files in a datastore
  if strcmp(opt.data_type, 'frames')
    filepath = [fileparts(data_path) filesep];
    allfiles = dir([filepath '*' opt.src_string '*']);
    T = size(allfiles,1);
    sz = size(imresize(imread([filepath allfiles(1).name]),opt.spatial_scale));
    %Store the file links to the raw data
    %data.raw = datastore(strcat(filepath, {allfiles.name}),'FileExtension','.tif','Type', 'image');
    data.raw = strcat(filepath, {allfiles.name});
    raw_stack_done = 0; if exist(get_path(opt,'raw_virtual_stack'),'file'), raw_stack_done = 1; end;
    if raw_stack_done, data.raw_stack.Y = chomp_data(get_path(opt,'raw_virtual_stack')); 
    else
      data.raw_stack.Y = chomp_data(get_path(opt,'raw_virtual_stack'), imread([filepath allfiles(1).name]), ...
        'number_format','uint16','timestamp', opt.timestamp, 'prefix',opt.file_prefix);
    end
    %Save the preprocessed stack
    data.proc_stack.Y = zeros([sz(1), sz(2), T]); % Image stack
    im_cur = single(imread([filepath allfiles(1).name]));
    data.proc_stack.Y(:, :,1) = imresize(im_cur,opt.spatial_scale,'bilinear');
    for i2 = 2:T
      im_cur = single(imread([filepath allfiles(i2).name]));
      data.proc_stack.Y(:, :,i2) = imresize(im_cur,opt.spatial_scale,'bilinear');
      if ~raw_stack_done, append(data.raw_stack.Y, im_cur, 'number_format','uint16'); end
    end
  elseif strcmp(opt.data_type, 'frames_virtual')
      filepath = [fileparts(data_path) filesep];
      allfiles = dir([filepath '*' opt.src_string '*']);
      T = size(allfiles,1);
      sz = size(imresize(imread([filepath allfiles(1).name]),opt.spatial_scale));
      %Store the file links to the raw data
      %data.raw = datastore(strcat(filepath, {allfiles.name}),'FileExtension','.tif','Type', 'image');
      %Old matlab versions don't support datastore, let's just not use it
      %for now
      data.raw = strcat(filepath, {allfiles.name});
      raw_stack_done = 0; if exist(get_path(opt,'raw_virtual_stack'),'file'), raw_stack_done = 1; end;
      if raw_stack_done, data.raw_stack.Y = chomp_data(get_path(opt,'raw_virtual_stack')); 
      else
        data.raw_stack.Y = chomp_data(get_path(opt,'raw_virtual_stack'), imread([filepath allfiles(1).name]), ...
        'number_format','uint16','timestamp', opt.timestamp, 'prefix',opt.file_prefix);
      end
      
      Ytmp = zeros([sz(1:2),100]);
      %Minibatch read and dump to the matfile variable
      s2 = 0; charcount = 0;
      for i2 = 1:T
        im_cur = imread([filepath allfiles(i2).name]);
        if (~raw_stack_done) && (i2>1), data.raw_stack.Y(:,:,end+1) = im_cur; end
        Ytmp(:,:,mod(i2,100)+floor(i2/100)*100) = imresize(single(im_cur),opt.spatial_scale,'bilinear');
        if (mod(i2,100) == 0) || (i2 == T)
          if s2==0
            data.proc_stack.Y = chomp_data(get_path(opt,'virtual_stack'),Ytmp(:,:,1:(mod(min(((s2+1)*100),T)-1,100)+1)),...
              'timestamp', opt.timestamp, 'prefix',opt.file_prefix, 'number_format','single');
          else
            append(data.proc_stack.Y, Ytmp(:,:,1:(mod(min(((s2+1)*100),T)-1,100)+1)));
          end
          s2 = s2+1;
          if opt.verbose
            if charcount>0, for c1 = 1:charcount, fprintf('\b'); end; end
            charcount = fprintf('Reading images and creating virtual stack... %d/%d',min(s2*100,T), T);
          end
        end
      end
  end % data reading
  
  
  if opt.verbose
    fprintf('\nPreprocessing image stack...');
  end
  
  % median filter the data in time before sub-sampling to reduce shot noise
  if opt.denoise_medianfilter
    data.proc_stack.Y = medfilt3(data.proc_stack.Y, [1,1,5]); 
  end
  
  %Downsample in time (average in subsequent time windows) %TODO: this is
  %quite inefficient
  szProc = chomp_size(data.proc_stack,'Y');
  if opt.time_scale < 1
    T = floor(T *opt.time_scale);
    for t1 = 1:T
      tmp = data.proc_stack.Y(:,:,ceil((t1-1)/opt.time_scale)+1); % take single samples instead of mean over time windows
      %tmp = mean(data.proc_stack.Y(:,:,ceil((t1-1)/opt.time_scale)+1:floor(t1/opt.time_scale)),3);
      data.proc_stack.Y(:,:,t1) = tmp;
    end
    %Reset dimensionality of the stacks
      data.proc_stack.Y(:,:,T+1:end) = [];
  end
  
  % Load (potential) preproc-supplied information
  if ~isempty(opt.spatial_gain) && numel(opt.spatial_gain)==1 && opt.spatial_gain==1    
    tmp = load([opt.root_folder, fileparts(opt.data_path) '/spatial_gain.mat']); % loads 'spatial_gain' image
    opt.spatial_gain = imresize(single(tmp.spatial_gain),opt.spatial_scale,'bilinear');
  end
  if opt.stretch_factor == 1
    opt.stretch_factor = single(csvread([opt.root_folder, fileparts(opt.data_path) '/stretch_factor.csv'])); % get stretch factor
  end 
  if opt.stretch_factor ~= 0 % Divide out the stretch that was added for numerical reasons
    data.proc_stack.Y = data.proc_stack.Y./opt.stretch_factor;
  end
  if opt.spatial_gain_undo_raw % If 1, data.proc_stack will be multiplied by the supplied spatial gain (keeps only photon count part from preproc)
    data.proc_stack.Y = data.proc_stack.Y.*opt.spatial_gain;
  end
  
  
  if opt.log_transform
    data.proc_stack.Y = log(data.proc_stack.Y+1.);
  end
  
%   % Find the "mode" (a bit smoothed) and remove it [If we still whiten,
%   % this doesn't matter, but this should be enough for preproc2P without
%   % whitening!
%   [tmp_counts,tmp_edges] = histcounts(reshape(data.proc_stack.Y(:,:,:),[],1), 2600); % 2600 bins in uin16 range is a 25 bin width
%   [~,tmp_whichbin] = max(tmp_counts(6:end)); % Ignore the first few bins, likely missing data
%   tmp_mode = tmp_edges(tmp_whichbin+5);
%   for t1 = 1:T
%     data.proc_stack.Y(:,:,t1) = data.proc_stack.Y(:,:,t1) - tmp_mode;
%   end
%   fprintf('\nSubtracted data mode of %0.2f ...', tmp_mode);
  
  %Get mean image of the raw data
  szRaw = chomp_size(data.raw_stack,'Y');
  y_orig = data.raw_stack.Y(:,:,1)./T;
  for i1 = 2:szRaw(3)
    y_orig = y_orig + data.raw_stack.Y(:,:,i1)./szRaw(3);
  end
 
  
    %Get mean image and variance, for cycle for virtual stack handling
  y = data.proc_stack.Y(:,:,1)./T;
  for i1 = 2:T
    y = y + squeeze(data.proc_stack.Y(:,:,i1))./T;
  end
  
  %Get the variance 
  if T>1
    V = zeros(size(y));
    for i1 = 1:T
      V = V + (squeeze(data.proc_stack.Y(:,:,i1))-y).^2./T;
    end
  else
    V = ones(size(y));
  end
  
  % Apply normalizing filters to mean image for visualization purposes and
  % to get "mean spatial mean (A) and mean spatial variance (B)"
  if T>1
    [y, A, B] = normal_img(single(y), opt.smooth_filter_mean , opt.smooth_filter_var ,V);
  else
    [y, A, B] = normal_img(single(y), opt.smooth_filter_mean , opt.smooth_filter_var);
  end
  opt.A = A;
  opt.B = B;
  
  
  
  %Apply whitening to the whole stack (whitening with respect to the
  %smoothing filter sizes set, not on a pixel-by-pixel level!)
  if opt.whiten
    data = whiten_proc(opt, data);
  elseif opt.subtract_background
  % Remove the remaining joint background value from all pixels at all
  % frames
    bg_level = get_hist_mode(opt.A,500);
    data.proc_stack.Y(:, :, :) = data.proc_stack.Y(:, :, :) - bg_level;
    fprintf('\nSubtracted background level of %0.2e ...', bg_level);
    opt.background_level = bg_level;
  end
  
  
  % Make sure that all values are positive to use NMF-like techniques
  if strcmp(opt.learn_decomp,'NMF')
    data.proc_stack.Y(:, :, :) = data.proc_stack.Y(:, :, :) - min(min(min(data.proc_stack.Y(:, :, :))));
  end
  
  %Get pixelwise cumulants of the whitened data
  cumulants_pixelwise = cell(4,1);
  for mom1 = 1:4
    cumulants_pixelwise{mom1} = zeros(size(y));
  end
  for i1 = 1:T
    V = V + (squeeze(data.proc_stack.Y(:,:,i1))-y).^2./T;
    for mom1 = 1:4 % Calculate raw moments first
      cumulants_pixelwise{mom1} = cumulants_pixelwise{mom1} + (squeeze(data.proc_stack.Y(:,:,i1)).^mom1);
    end
  end
  
  if opt.zeros_ignore % Treat zeros as missing data, scale the expected moments up accordingly
    opt.zeros_count_pixelwise = 1.*sum(data.proc_stack.Y==0,3);
    if (opt.zeros_count_pixelwise(:)./(1.*T))>(1-opt.zeros_min_nonzero_frac_needed), warning('CHOMP:extract_data:opt.zeros_ignore=1, but there are pixels with very low non-zero count, statistics may be unreliable for those');end
    for mom1 = 1:4
      cumulants_pixelwise{mom1} = cumulants_pixelwise{mom1}.*(T./(T-opt.zeros_count_pixelwise));
      % But if too many zeros, just keep them as zero (cannot get
      % statistics for those)
      cumulants_pixelwise{mom1}((opt.zeros_count_pixelwise(:)./T)>(1-opt.zeros_min_nonzero_frac_needed)) = 0;
    end
  end

  opt.cumulant_pixelwise = raw2cum_kstat(cumulants_pixelwise, T); % Store the cumulant estimates pixelwise up to 4th order.
  
  if opt.standardise_cumulants
    opt.cumulant_pixelwise = cum2standardised(opt.cumulant_pixelwise);
  end
  
%   %Fix edge_effect problems later %TODO
%   edge_effect = conv2(ones(size(y)),ones(opt.m),'same');
  
  



  inp = chomp_input(opt,data, mean(data.proc_stack.Y,3), y_orig,V); %The input class that stores raw and preproc data
  
  
   %let the user create a mask over the image to limit the processed area
  if opt.mask
    if ~isempty(opt.mask_image) 
        inp.UserMask = opt.mask_image;
    else
      imshow(y);
      inp.UserMask = roipoly(mat2gray(y));
    end
  end
  
  save(intermediate_path, 'inp','-v7.3');
  
  

end

 
  
  clearvars -except 'opt'

end
  
  
  

