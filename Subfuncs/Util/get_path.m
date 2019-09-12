function output = get_path( opt,varargin)
%GET_PATH Gets the path for certain important files, by default the intermediate input data file, with different secondary options you can modify that

if nargin==1
  if ~isempty(opt.timestamp)
    output = [opt.input_folder opt.file_prefix '_' opt.timestamp '.mat'];
  else %Get file with newest timestamp
    output = dir([opt.root_folder opt.input_folder opt.file_prefix '_*T*.mat']);
    [~, tmp] = sort([output.datenum]);
    output = [opt.input_folder output(tmp(end)).name];
  end
else
  switch varargin{1}
    case 'output_iter'
      %Require which output iteration as varargin{2}
      output = [opt.output_folder opt.file_prefix '_' opt.timestamp '_iter_' num2str(varargin{2}) '.mat'];
    case 'precomputed'
      if ~opt.diag_cumulants % we need the full tensor
        output = [opt.precomputed_folder 'Precomputed_shift_tensor_window_' num2str(opt.m) '_moment_' num2str(opt.mom) '.mat'];
      else % we only need the 1st order shift tensor
        output = [opt.precomputed_folder 'Precomputed_shift_tensor_window_' num2str(opt.m) '_moment_' '1' '.mat'];
      end
    case 'virtual_stack'
      output = [opt.input_folder opt.file_prefix '_' opt.timestamp '_virtual_stack.chd'];
    case 'raw_virtual_stack'
      output = [opt.input_folder opt.file_prefix '_virtual_stack_raw.chd'];
    case 'raw_stabilized_frames'
      if numel(varargin) == 1 %return folder
        output = [opt.input_folder opt.file_prefix '_stabilized' filesep];
      else %return frame to write to
        output = [opt.input_folder opt.file_prefix '_stabilized' filesep opt.file_prefix '_stabilized_' sprintf('%.5d',uint16(varargin{2})) '.tif'];
      end
    case 'results'
      output = [opt.results_folder opt.file_prefix '_' opt.timestamp '_results.mat'];
    case 'results_image'
      output = [opt.results_folder opt.file_prefix '_' opt.timestamp '_results_image.png'];
    otherwise
       error('CHOMP:util:nopath', 'The type of path you want get_path() to return is not implemented');
  end
end

output = [opt.root_folder output]; %Add the root folder in case of sshfs work

end

