function c=STCDOindexMC(LGD,r,N0,tau,Tj,annuity)
%%STCDOINDEX computes the single-tranche CDO index for given simulated 
% default times.
%
%   Input:
%       LGD (1 x 1 double): loss given default
%       r (1 x 1 double): determ. interest rate
%       N0 (1 x 1 double): notional
%       tau (K x M double): simulated default events of K firms
%       Tj (1 x n or n x 1 double): resettlement dates in years
%       annuity (1 x 1 double): annuity in years, e.g., .25 for quarterly
%
%   Output:
%       c (1 x 1 double): STCDO index
%
%   Usage:
%       c=STCDOindex(a,d,LGD,r,N0,tau,Tj,annuity)
%
%   See also:
%       defaultTime, distanceToDefault, brownianMotions

    if Tj(1)~=0
        Tj=[0;Tj(:)];
    else
        Tj=Tj(:);
    end

    tauShape=size(tau);
    tau=reshape(tau,[1,tauShape]);
    ind1=tau>Tj; % n x K x M

    % Outstanding index notional
    ZIt=N0.*sum(ind1,2);

    % STCDO spread
    bt=exp(-r.*Tj(2:end));
    nom=LGD.*sum(bt.*mean((ZIt(1:end-1,:,:)-ZIt(2:end,:,:)),3),1);
    denom=annuity.*sum(bt.*mean(ZIt(2:end,:,:),3),1);
    c=nom./denom;
end