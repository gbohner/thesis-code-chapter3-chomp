classdef chomp_options < handle
  %OPTIONS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties (SetObservable, GetObservable, AbortSet)
    root_folder = ''; % Can set a root folder in case of remote work (sshfs or runs from different users/systems mac vs linux directory structure). All other folders below will be calculated relative from the root folder
  end
  
  properties
    %Setup folder structure
     code_path = [fileparts(mfilename('fullpath')) filesep]; %Package directory
     data_path = 'default_path'; %Input data file (stack, or initial frame)
     input_folder = './tmp/input/'; %Store preprocessed input data, if you change it, use full path
     output_folder = './tmp/output/'; %Output folder, if you change it, use full path
     precomputed_folder = './tmp/precomputed/'; % Stores precomputed tensors
     results_folder = './tmp/results/'; %Store the extracted ROIs and timeseries
     file_prefix = 'test'; %Prefix for file names
 
     % Model setup
     m = 17; % Basis function size in pixels
     NSS = 2; % Number of object types
     KS = 4; % Dimensionality of space per object type (i.e. number of basis functions per object type)
     init_model = {'filled', 'pointlike'}; % 'filled', 'donut', 'pointlike', \\ %TODO: 'supervised', 'multi'
     init_W = [];

     % Data extraction and preprocessing
     stabilize = 1;
     spatial_scale = 1; % Rescale data spatially (so that cell size matches basis function size)
     time_scale = 1; % Rescale data temporally
     log_transform = 0; % After spatially and temporally scaling the data, do x<-log(x+1) transform
     whiten = 0;
     cell_pixel_fraction = []; % Estimate (upper bound) of the fraction of pixels that belong to cells
     smooth_filter_sigma = 0.4; % Gaussian low-pass smoothing sigma (in pixels)
     smooth_filter_mean = 0.1; % = 0.3; %smoothing filter size for mean image (smooth over a ~3x3 area by default)
     smooth_filter_var = 4; % = m/5; %smoothing filter size for variance
     subtract_background = 1; % If true, subtract the mode from all images.
     background_level = 0;
     denoise_medianfilter = 1; % If true, applies median filter denoising
     data_type = 'frames_virtual'; %Input data type (frames / stack / json / matxyt)
     src_string = 'Ch2_*'; %in case of loading multiple frames from a directory, look for this substring to load files (choose channel eg)
     mask = 0; % Set if the region of interest is only part of the image stack.
     mask_overwrite = 0; %If a previous intermediate file already has mask, do you want the to create a new one (1) or automatically use the old one (0).
     mask_image = []; % you can input your own binary mask image if needed
     mom = 2; %Number of moments used
     A % Mean image for spatial whitening
     B % Variance for spatial whitening
     cumulant_pixelwise = {}; % Store pixelwise cumulants up to order 4 for visualisation and later use.

     % Learning parameters
     niter = 4; % number of iterations
     init_iter = 0; %Set the initial iteration. If not 0, search for the file with appropriate name
      %     relweight = 10; % weighting between importance of covariance / mean (automatically set to 'optimal' value in Shared_main/extract_coefs.m)
     cells_per_image = 30; % the maximum number of objects to infer
     learn   = 1; % do learning?
     spatial_push = @(grid_dist)logsig(0.5*grid_dist-floor(7/2-1)); % Specified distance based function (set as [] if not desired)
     learn_decomp = 'COV_RAW'; % COV_RAW, COV, HOSVD, NMF or MTF (MTF not implemented yet, %TODO - write R wrapper to use Kahn2015 code)
     diag_tensors = 0; % Use only b_k^r as reconstructions of r order, not b_k1*b_k2*...*b_kr types
     diag_cumulants = 0; % Use only y_l^r pixelswise moments for reconstruction, not y_l1*y_l2*...y_lr full patch tensors
     diag_cumulants_offdiagonly = 0; % Use only y_l^r pixelswise moments for reconstruction, not y_l1*y_l2*...y_lr full patch tensors
     diag_cumulants_extradiagonal = 0; % If 1, then adds extra dimension to WY by creating strictly diagonal higher order filters.
     W_weight_type = 'uniform'; % uniform / decomp  % Type of basis function weighting during learning
     W_weights = []; % Basis function weights
     W_addflat = 0; %Adds an all ones basis function that is used in reconstruction but not in likelihood calculations?
     W_force_round = 0; % If 1, it forces the W bases to be zero outside of an opt.m-diameter circle (essential no corners)
     mom_weights = []; % Moment weights
     local_peaks = 0; % If 1, instead of doing global zscore per dL_mom, it uses local context to renormalise dL_mom and select next reconst location
     local_peaks_size = []; % 3-element vector, used as [filter_size, circle_radius, inner_radius] in [~,mask]=transform_inds_circ(0,0,150,opt.local_peaks_size(1),opt.local_peaks_size(2),opt.local_peaks_size(3));
     local_peaks_gauss = 0; % If larger than 0, serves as the lengthscale of a Gaussian filter (see extract_coefs)
     blank_reconstructed = 0; % If 1, treats reconstructed areas as "missing data" instead of attempting to reconstruct the residuals there
     standardise_cumulants = 0; % If 1, uses the standardised cumulants instead of raw cumulants
     spatial_gain = []; % If scalar, find and load a spatial gain matrix (the result of preproc) (otherwise supply the matrix)
     spatial_gain_undo_raw = 0; % If 1, data.proc_stack will be multiplied by the supplied spatial gain
     stretch_factor = 0; % If 1 - load the stretch factor from preproc. If >0, but ~=1, keep it as scalar
     spatial_gain_renormalise_lik = 1; % If spatial gain is supplied, use it to renormalise likelihood
     reconst_upto_median_WY = 0; % If 1, for each dimension of WY, gets the median (NOT IMPLEMENTED)
     zeros_ignore = 0; % If 1, 0 values are treated as missing data for all intents and purposes
     zeros_count_pixelwise = []; % Store number of zeros per pixel for statstics later
     zeros_min_nonzero_frac_needed = 0.1; % Minimum fraction of nonzero element to get actual statistics
     
    % Extracting ROIs
     ROI_type = 'quantile_dynamic_origsize';
     ROI_params = [0.4];


     % Misc parameters
     fig = 1; %How much to visualize - levels 0, 1, 2
     verbose = 1; %How much progress to show in text levels 0, 1, 2
     cleanup = 1; % 1 - close all open files, 2 - delete all intermediate files (2 is not impletemented yet)

     timestamp % Timestamping
     
     
  end
  
  properties (Dependent)
    Wblocks
  end
  
  methods
    function obj = chomp_options(varargin)
      %Object constructor
      
      %Construct from a struct
      if nargin==1
        if isa(varargin{1},'struct')
          fns = fieldnames(varargin{1});
          for i1 = 1:numel(fns)
            try
              obj.(fns{i1}) = varargin{1}.(fns{i1});
            catch ME
              rethrow(ME); %TODO better error message
            end
          end
        else
          error('CHOMP:chomp_options:bad_constructor','Wrong class constructor call for chomp_options');
        end
      end
      
          
      %Construct from ('field', 'value') argument pairs
      i1 = 1;
      while i1<nargin
        obj.(varargin{i1}) = varargin{i1+1};
        i1 = i1+2;
      end
      
      %Compute the derived properties
      obj = obj.derive_from_m(); 
      
      obj = obj.assert();
    end
    
    function obj = assert(obj)
      %Check for specific parameters to be in the correct format;
      if ~iscell(obj.init_model), obj.init_model = {obj.init_model}; end
      assert(numel(obj.init_model)==obj.NSS, 'CHOMP: Object type # discrepency');
    end
    
    
    function obj = derive_from_m(obj)
      %On construct, makes sure that certain properties are set correctly
      %in relation to basis function (i.e. expected cell) size
%       obj.smooth_filter_mean = obj.m;
%       obj.smooth_filter_var = obj.m;
      % obj.spatial_push = @(grid_dist)logsig(0.5*grid_dist-floor(obj.m/sqrt(2)-1)); %@(grid_dist, sharp)logsig(sharp*grid_dist-floor(sharp*2*obj.m/2-1));
    end
    
    function s = export_struct(obj, varargin)
      p = properties(obj);
      for i1 = 1:numel(p)
        s.(p{i1}) = obj.(p{i1});
      end
    end

    function blocks = get.Wblocks(obj)
      blocks = cell(obj.NSS, 1);
      for type = 1:obj.NSS
        blocks{type} = ((type-1)*obj.KS+1):(type*obj.KS);
      end
    end
    
    function set.Wblocks(obj, val)
      %Just to suppress errors coming from no set method, it doesn't do
      %anything.
    end
  end
  
  
  
  
end

