% Run all of neurofinder files with given timestamps and given (custom)
% options

cur_timestamp = datestr(now(),30);

input_opts = struct();
input_opts.timestamp = cur_timestamp;

input_opts.whiten = 1;
input_opts.mom = 1;
