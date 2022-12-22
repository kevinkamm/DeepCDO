function [c,varargout]=STCDOspread(a,d,r,Lt,Tj)
%%STCDOSPREAD computes the single-tranche CDO spread for given simulated 
% default times.
%
%   Input:
%       a (1 x 1 x p double): tranche attachment point
%       d (1 x 1 x p double): tranche detachment point
%       Lt (n x M double): simulated portfolio loss
%       Tj (1 x n or n x 1 double): resettlement dates in years 
%
%   Output:
%       c (p x 1 double): STCDO spread for a-d tranche
%
%   Usage:
%       c=STCDOspread(a,d,LGD,r,N0,tau,Tj,annuity)
%
%   See also:
%       portfolioLossMC, portfolioLossTheta, portfolioLossEM

    if Tj(1)~=0
        Tj=[0;Tj(:)];
    else
        Tj=Tj(:);
    end
    annuity=diff(Tj,1,1);

    % Outstanding notional
    Zt=max(d-Lt,0)-max(a-Lt,0);

    % STCDO spread
    bt=exp(-r.*Tj(2:end));  % discount factor
    nom=sum(bt.*mean(Zt(1:end-1,:,:)-Zt(2:end,:,:),2),1);  % protection leg
    denom=sum(annuity.*bt.*mean(Zt(2:end,:,:),2),1);  % fee leg
    c=squeeze(nom./denom);  % spread

    if nargout>1
        varargout{1}=Zt;
    end
    if nargout>2
        % tranche losses
        Yt=mean(max(Lt-aPoint,0)-max(Lt-dPoint,0),2);
        varargout{2}=Yt;
    end
end