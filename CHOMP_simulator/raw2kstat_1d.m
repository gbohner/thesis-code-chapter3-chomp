function out = raw2kstat_1d( inp, T, dim )
%K_STATISTIC Converts raw moments to cumulants for 1D distributions for data
%matrix inp, where raw moments are stored along dimension dim

inp = inp*T;

%Put the moment dimension first
dims = 1:ndims(inp);
dims(1) = dim;
dims(dim) = 1;
inp = permute(inp, dims);

%Convert the raw moments into kstatistic estimate of cumulants
szInp = size(inp);
inp = reshape(inp,szInp(1),[]);
out = inp;

out(1,:) = inp(1,:)./T;
if szInp(1)>=2
  out(2,:) = (T*inp(2,:) - inp(1,:).^2)/(T*(T-1));
end
if szInp(1)>=3
  out(3,:) = (T^2*inp(3,:) -3*T*inp(2,:).*inp(1,:) + 2* inp(1,:).^3) / (T*(T-1)*(T-2));
end
if szInp(1)>=4
  out(4,:) = ( T.^2*(T+1)*inp(4,:) - 4 * T * (T+1) *inp(3,:).*inp(1,:) - 3*T*(T-1)*inp(2,:).^2 + 12*inp(2,:).*inp(1,:).^2 - 6*inp(1,:).^4) / (T*(T-1)*(T-2)*(T-3));
end

%Permute the dimensions back
out = reshape(out,szInp);
out = ipermute(out,dims);
end

