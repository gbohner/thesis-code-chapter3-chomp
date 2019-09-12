function y = create_data( I, T, L, X, B, eta, baseline)
%CREATE_DATA Simulates from the generative model with dimensions
%I(1)x...xI(d)xT, with object locations L and object signal described by
%BxX, corrupted by gaussian noise level eta

y = randn([I, T])*eta+baseline;
d = floor(size(B,1)/2);
sb = size(B); 
sb = num2cell(sb);

for l = 1:size(L,1)
    for p = 1:size(L,2)
        l_patch{p} = (L(l,p)-d):(L(l,p)+d);
    end
    [ valid_inds, cuts ] = mat_boundary(I, l_patch{:});
        
    y(valid_inds{:},:) = y(valid_inds{:},:) + mply(B,squeeze(X(l,:,:)),1);


end

