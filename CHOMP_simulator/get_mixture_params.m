function F = get_mixture_params( x, cums, varargin )
%MIXTURE_NONLIN_SYST Gets a set of cumulants as input, and outputs params
%for a mixture of gaussians that implement those cumulants

if nargin>2
  output_exp_cums = varargin{1};
else
  output_exp_cums = 0; %outputs the value needed for fsolve
end
  

% Try to solve with just mixture of 2 gaussians, up to 4 cumulants
if numel(cums)<4, cums(numel(cums)+1:4) = 0; end

% specify parameters
x = mat2cell(x,ones(numel(x),1));
[mu1, mu2,sig1,sig2,p] = deal(x{:}); % set the 5dim parameter vector

mus = [mu1, mu2];
sigs = [sig1,sig2];
ps = [p.^2, 1-p.^2];

F = zeros(numel(cums,1));

%Formula's from Jin Wang 2000, Modeling and Generating Daily Changes in Market Variables Using A Multivariate Mixture of Normal Distributions
F(1) = ps * mus'; %mean formula
F(2) = ps * (sigs'.^2 + mus'.^2) - F(1).^2; %variance formula
F(3) = 1/(sqrt(F(2)).^3) .*  ( ps * ( (mus'-F(1)).*(3*sigs'.^2 + (mus'-F(1)).^2) )); %skewness formula
F(4) = 1/(F(2).^2) .* ( ps * (3 * sigs'.^4 + 6*((mus'-F(1)).^2).*(sigs'.^2) + (mus'-F(1)).^4)); % kurtosis forumla

if ~output_exp_cums
  F = F(:) - cums(:); % set the condition for fsolve
%   % Try not to control skewness
%   F(3) = 0;
end
  


end

