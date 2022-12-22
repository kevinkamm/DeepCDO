function LtTheta=portfolioLossTheta(t,Tj,x,dMt,beta,rho,v0,LGD,theta)
    M=size(dMt,2);
    d=length(x);
    dx=(x(end)-x(1))/(d-1);

    % remove bd points
    x=x(2:end-1);
    d=length(x);

    D=speye(d);
    Dx=spdiags(([-1 0 1]./(2.*dx)).*ones(d,1),-1:1,d,d);
    Dxx=spdiags(([1 -2 1]./(2.*dx.^2)).*ones(d,1),-1:1,d,d);

    N=length(t);
    dt=t(end)/(N-1);

    nTheta=(N-1)/(length(Tj)-1);

    LtTheta=zeros(length(Tj),M);
    LtTheta(1,:)=LGD.*(1-sum(v0,1).*dx);
    
    tTemp=linspace(0,Tj(2),nTheta+1);
    vtTheta=v0;
    for Ti=1:1:length(Tj)-1
        utTheta=determThetaScheme(D,Dx,Dxx,vtTheta,tTemp,rho,beta,theta);
        vtTheta=updateInitialDatum(utTheta,rho,dMt(Ti,:),x);
        LtTheta(Ti+1,:)=LGD.*(1-sum(vtTheta,1).*dx);
    end

end
function vt=determThetaScheme(D,Dx,Dxx,v0,t,rho,beta,theta)    
    N=length(t);
    dt=t(end)./(N-1);

    vt=v0;
    
    % notice that Matlab2022a is faster than Thomas Algo
    % Maybe cyclic reduction or Fourier based methods are faster
    if theta>0
        %% Rannacher start-up, implicit Euler for 4 half steps
        Lh=decomposition(D-(dt./2).*((1-rho).*Dxx-beta.*Dx));
        for i=1:1:4
            rh=vt;
            vt=Lh\squeeze(rh);
        end
        %% Theta scheme
        Lh=decomposition(D-theta.*dt.*((1-rho).*Dxx-beta.*Dx));
        for i=3:1:N-1
            rh=vt+(1-theta).*dt.*((1-rho).*Dxx-beta.*Dx)*vt;
            vt=Lh\rh;
        end
    else
        %% Explicit Euler
        Rh=dt.*((1-rho).*Dxx-beta.*Dx);
        for i=1:1:N-1
            vt=vt+Rh*vt;
        end
    end
end
function vtShift=updateInitialDatum(vt,rho,dM,x)
%%UPDATEINITIALDATUM approximates the dirac initial datum in x0 by the integral 
% of linear splines on a given space grid x.
    x=x(:);
    d = length(x);
    M=length(dM);
    sz=size(vt);
    if sz(end)==1
        vt=reshape(vt,d,1).*ones(1,M);
    else
        vt=reshape(vt,d,M);
    end
    dM=sqrt(rho).*reshape(dM,1,M);

    vtShift=zeros(size(vt));
    parfor wi=1:M
        temp=griddedInterpolant(x,vt(:,wi),'spline');
        vtShift(:,wi)=temp(x-dM(1,wi));
    end

    xNeg=x<=0;
    vtShift(xNeg,:)=0;
end
