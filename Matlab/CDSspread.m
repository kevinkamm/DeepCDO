function c=CDSspread(LGD,r,tau,Tj,annuity)
%%CDSSPREAD computes the CDS spread for given simulated default times.
%
%   Input:
%       LGD (1 x 1 double): loss given default
%       r (1 x 1 double): determ. interest rate
%       tau (K x M double): simulated default events of K firms
%       Tj (1 x n or n x 1 double): resettlement dates in years
%       annuity (1 x 1 double): annuity in years, e.g., .25 for quarterly
%
%   Output:
%       c (K x 1 double): CDS spreads for K firms
%
%   Usage:
%       c=CDSspread(LGD,r,tau,Tj,annuity)
%
%   See also:
%       defaultTime, distanceToDefault, brownianMotions
    
    % Check arguments
%     arguments
%         LGD (1,1) double
%         r (1,1) double
%         tau (:,:) double
%         Tj (1,:) double
%         annuity (1,1) double
%     end
    
    if Tj(1)~=0
        Tj=[0;Tj(:)];
    else
        Tj=Tj(:);
    end
    tauShape=size(tau);
    tau=reshape(tau,[1,tauShape]);

    ind1=Tj>tau;
    ind2=Tj<tau;

    % negative spread
%     nom = LGD.*mean(sum(exp(-r.*Tj(2:end)).*(ind1(1:end-1,:,:)-ind1(2:end,:,:)),1),3);

    % positive spread
    bt=exp(-r.*Tj(2:end));
    nom = LGD.*sum(bt.*mean(ind1(2:end,:,:)-ind1(1:end-1,:,:),3),1);

    denom = sum(annuity.*bt.*mean(ind2(2:end,:,:),3),1);
    c=squeeze(nom./denom);
end