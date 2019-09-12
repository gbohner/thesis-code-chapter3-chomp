function out = cum2standardised(inp)
%CUM2STANDARDISED Standardises pixelwise cumulants (divide mom1>2 with
%(var).^(mom1/2); %inp is cell array, where each cell is the mom1-th
%cumulant

out = inp;
for mom1=3:length(out)
  out{mom1} = out{mom1}./(out{2}.^(mom1./2.));
  
  % This is for clarity for 0-variance pixels
  out{mom1}(out{2}<=0)=0;
end
end

