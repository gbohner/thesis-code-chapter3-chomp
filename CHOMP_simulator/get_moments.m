function M = get_moments( Y, B, mom, varargin )
%GET_MOMENTS Gets the moment estimators of the diagonal coefficients that live in
%the space of basis vectors

if nargin > 3, mom_type = varargin{1}; else mom_type='scaled_identity'; end
if nargin > 4, cumulant = varargin{2}; else cumulant=1; end

num_ss = size(B,ndims(B));

T = size(Y,ndims(Y));
I = size(Y); I = I(1:(ndims(Y)-1));
%Get projections:
P = zeros(num_ss,prod(I),T);
for i1 = 1:num_ss
    Bdims = cell(1,ndims(B)); [Bdims{1:end-1}] = deal(':');
    Bdims{end} = i1;
    P(i1,:,:) = reshape(convn(padarray(Y,(size(B,1)-1)/2*ones(1,numel(I)),'circular','both'),flipud(B(Bdims{:})),'valid'),1,prod(I),T); %ss, L, T
end
% 

switch mom_type
    case 'scaled_identity'
        %Get identity moment estimates
        M = zeros(mom,prod(I));
        Wss = ones(1,num_ss); %weighting of subspace directions if we'd sum them up (case everything is mu * I)
        for m1 = 1:mom
            M(m1,:) = Wss*mean(P.^m1,3);
        end
        if cumulant
          M = raw2cum(M,1);
        end
    case 'diagonal'
        %Get diagonal moment estimates
        M = zeros(mom,num_ss,prod(I));
        for m1 = 1:mom
            M(m1,:,:) = mean(P.^m1,3);
        end
        %Turn it into cumulants:
        if cumulant
          M = raw2kstat_1d(M,T,1);
          %M=raw2cum(M,1);
        end
        M = reshape(M,mom*num_ss,[]); %all params * all_locations
    case 'full'
        %Get full moment estimates (atm works for two bases only and not
        %cumulants)
        elems = 0; for i1 = 1:mom, elems = elems + num_ss.^i1; end %all parameters we gotta store
        M = zeros(elems,prod(I));
        cur_elems = 0;
        for m1 = 1:mom
          indset = dec2bin((0:(2^mom-1))')-'0'; % all binary indicators
          for r1 = 1:size(indset,1)
            M(cur_elems+r1,:) = mean(prod(P(indset(r1,:)+1,:,:),1),3); %take the product of the corresponding dimensions
          end
          cur_elems=cur_elems+r1;
        end
end


% % Get proper moment estimates
% M = cell(mom);
% for m1 = 1:mom
%     M{m1} = zeros(num_ss.^m1,prod(I));
%     M{m1}(:,i1) = rowfun(@(inp)myouter(inp,m1),reshape(P,size(P,1),:));
% end
% 
% function output = myouter(P_l,curmom)
%     P_l = reshape(P_l,num_ss,T)';
%     if curmom == 1, output = mean(P_l,1); return, end
%     for cur1 = 2:curmom
%         P_lmom = mply
    
end