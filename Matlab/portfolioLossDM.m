function LtTheta=portfolioLossDM(t,Tj,x,dMt,beta,rho,v0,LGD)
    M=size(dMt,2);
    d=length(x);
    dx=(x(end)-x(1))/(d-1);

    % remove bd points
    x=x(2:end-1);
    d=length(x);

%     D=speye(d);
    Dx=spdiags(([-1 0 1]./(2.*dx)).*ones(d,1),-1:1,d,d);
    Dxx=spdiags(([1 -2 1]./(2.*dx.^2)).*ones(d,1),-1:1,d,d);



    LtTheta=zeros(length(Tj),M);
    LtTheta(1,:)=LGD.*(1-sum(v0,1).*dx);
    
    vtTheta=v0;
    B=expm(((1-rho).*Dxx-beta.*Dx).*(Tj(2)-Tj(1)));
%     B(B<1e-6)=0;
%     B=sparse(B);
    for Ti=1:1:length(Tj)-1
        utTheta=B*vtTheta;
        vtTheta=updateInitialDatum(utTheta,rho,dMt(Ti,:),x);
        LtTheta(Ti+1,:)=LGD.*(1-sum(vtTheta,1).*dx);
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