function out = raw2cum_kstat( inp, T )
%RAW2CUM Converts raw moments to cumulants for cell array inp, 
% where raw moment unnormalised sums are stored within the individual cells inp{1} = mean,
% inp{2} = second moment etc AND the individual entries of the matrices are
% independent

%Put the moment dimension first
out = inp;

if length(out)>=1
  out{1} = inp{1}./T;
end
if length(out)>=2
  out{2} = (T.*inp{2} - inp{1}.^2)./(T*(T-1));
  out{2}(out{2}<0) = abs(out{2}(out{2}<0)); % Correction 
end
if length(out)>=3
  out{3} = (T^2*inp{3} - 3*T*inp{2}.*inp{1} + 2*inp{1}.^3)./(T*(T-1)*(T-2));
end
if length(out)>=4
  out{4} = ((T^2)*(T+1)*inp{4} - 4*T*(T+1)*inp{3}.*inp{1} - 3*T*(T-1)*inp{2}.^2 +12*T*inp{2}.*(inp{1}.^2) - 6*inp{1}.^4)./(T*(T-1)*(T-2)*(T-3));
end

end

