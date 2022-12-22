function [LtMC,varargout]=portfolioLossMC(t,Tj,Wt,Mt,r,sigma,rho,x0,LGD)
%%PORTFOLIOLOSSMC

    % Simulate distance to default 
    Xt=distanceToDefault(t,Wt,Mt,r,sigma,rho,x0);
    
    % Default time equals inf_t X_t<=0
    tau=defaultTime(t,Xt);
    
    % Compute portfolio loss
    if Tj(1)~=0
        Tj=[0;Tj(:)];
    else
        Tj=Tj(:);
    end
    
    tauShape=size(squeeze(tau));
    tau=reshape(tau,[1,tauShape]);
    defaulted=tau<=Tj;
    
    % Portfolio loss evaluated at Tj
    N0=1/size(Wt,1);
    LtMC=squeeze(N0.*LGD.*sum(defaulted,2));
    
    if nargout>1
        varargout{1}=tau;
    end
end
function Xt=distanceToDefault(t,Wt,Mt,r,sigma,rho,x0)
%%DISTANCETODEFAULT computes the distance to default for given structural
% model with common Brownian motion Mt and individual firm Brownian motions
% Wt whose correlation is rho with parameters r,sigma and initial value x0.
%
%   Input:
%       t (1 x N double): time grid
%       Wt (K x N x M double): firm Brownian motions
%       Mt (1 x N x M double): common Brownian motion
%       r (1 x 1 double): determ. interest rate
%       sigma (1 x 1 double): volatility
%       rho (1 x 1 double): correlation
%       x0 (K x 1 x M double): initial value
%
%   Output:
%       Xt (K x N x M double): distance to default
%
%   Usage:
%       X=distanceToDefault(t,Wt,Mt,r,sigma,rho,x0)
%
%   See also:
%       brownianMotions, defaultTime

    Xt=x0+(r-sigma.^2./2)./sigma .* t +...
      sqrt(1-rho).*Wt+...
      sqrt(rho).*Mt;
end
function [T0,varargout]=defaultTime(t,Xt)
%%DEFAULTTIME computes the time of default for given distances of default.
%
%   Input:
%       t (1 x N double): time grid
%       Xt (K x N x M double): distance to default
%
%   Output:
%       T0 (K x M double): time of default
%
%   Usage:
%       T0=defaultTime(t,Xt)
%       [T0,Xt]=defaultTime(t,Xt): sets trajectories of Xt to zero after
%                                  default
%
%   See also:
%       brownianMotions, distanceToDefault

    [K,~,M]=size(Xt);
    d0=0;
    if nargout>1
        Yt=Xt;
    end
    T0=2.*t(end).*ones(K,M); % 2 * T, to make sure its not defaulted at T
    Xind=Xt<=d0;
    Xind2=sum(Xind,2);
    for iK=1:1:K
        for iW=1:1:M
            if Xind2(iK,1,iW)>0 %if only few defaults, then this speeds up the code
                indTi0=find(Xind(iK,:,iW)); % inf_t X_t^i<=0
                if ~isempty(indTi0)
                    T0(iK,iW)=t(indTi0(1));
                    if nargout>1
                        Yt(iK,indTi0(1):end,iW)=0;
                    end
                end
            end
        end
    end
    if nargout>1
        varargout{1}=Yt;
    end
end