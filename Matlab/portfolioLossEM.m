function LtEM=portfolioLossEM(t,Tj,x,Mt,beta,rho,v0,LGD)
    

    M=size(Mt,2);
    d=length(x);
    dx=(x(end)-x(1))/(d-1);

    % remove bd points
    x=x(2:end-1);
    d=length(x);
    
    Dx=spdiags(([-1 0 1]./(2.*dx)).*ones(d,1),-1:1,d,d);
    Dxx=spdiags(([1 -2 1]./(dx.^2)).*ones(d,1),-1:1,d,d);

    N=length(t);
    dt=t(end)/(N-1);

    nEM=(N-1)/(length(Tj)-1);

    % Compute portfolio loss
    if Tj(1)~=0
        Tj=[0;Tj(:)];
    else
        Tj=Tj(:);
    end

    LtEM=zeros(length(Tj),M);
    LtEM(1,:)=LGD.*(1-sum(v0,1).*dx);
    
    xNeg=x<=0;

    vtEuler=v0;

    for Ti=1:1:length(Tj)-1
        currdMt=diff(Mt(1+(Ti-1)*nEM:1+Ti*nEM,:),1);
        vtEuler=eulerMaruyama(Dx,Dxx,beta,rho,currdMt,vtEuler,dt);
        vtEuler(xNeg,:)=0;
        LtEM(Ti+1,:)=LGD.*(1-sum(vtEuler,1).*dx);
    end

end
function vt=eulerMaruyama(Dx,Dxx,beta,rho,dMt,v0,dt)
    [N,M]=size(dMt);
    N=N+1;

    sz0=size(v0);
    if sz0(end)==1
        vt=reshape(v0,[],1).*ones(1,M);
    else
        vt=reshape(v0,[],M);
    end

    for ti=1:1:N-1
        vt=vt-beta.*Dx*vt.*dt+Dxx*vt./2.*dt-sqrt(rho).*Dx*vt.*dMt(ti,:);
    end
end