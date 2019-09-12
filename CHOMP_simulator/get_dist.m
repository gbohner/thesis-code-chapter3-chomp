function [X, x1] = get_dist(samp_size, cums )
%GET_DIST Generates sampels from a mixture of two guassians such that the
%samples come from a distribution with the given cumulants

opts = optimoptions('fsolve','Display','off','MaxFunEvals',20000,'MaxIter',20000, ...
  'Algorithm','levenberg-marquardt');
x0 = [0, cums(1)*2, sqrt(cums(2))/100, sqrt(cums(2)), 0.5]';
x1 = fsolve(@(x)get_mixture_params(x, cums), x0, opts);

mu1 = x1(1);
mu2 = x1(2);
sig1 = x1(3);
sig2 = x1(4);
p = x1(5);

obj = gmdistribution([mu1; mu2],shiftdim([sig1.^2; sig2.^2],-2),[p.^2; 1-p.^2]);

X = random(obj,prod(samp_size));
X = reshape(X,samp_size);

%X = p * ((randn(samp_size)*sig1 + mu1)) + (1-p) * (randn(samp_size)*sig2 + mu2);

% %For diagnosis
% wanted_cums = cums
% expected_cums = get_mixture_params( x1, [0,0,0,0], 1 )
% true_cums = [mean(X(:)), var(X(:)), skewness(X(:)), kurtosis(X(:))]
% %[0 x1']
% close all;
% hist(X,64);
end

