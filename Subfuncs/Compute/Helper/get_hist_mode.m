function [out_mode, counts, edges] = get_hist_mode(inp, num_bins, ignore_first_n)
%GET_HIST_MODE Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
  ignore_first_n = 5;
end

[counts,edges] = histcounts(inp(:), num_bins);
      [~,tmp_whichbin] = max(counts((ignore_first_n+1):end)); % Ignore the first few bins, likely missing data
      out_mode = edges(tmp_whichbin+ignore_first_n);
end

