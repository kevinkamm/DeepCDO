function LtSM=portfolioLossSM(t,Tj,x,Mt,beta,rho,v0,LGD,order)
    

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

    LtSM=zeros(length(Tj),M);
    LtSM(1,:)=LGD.*(1-sum(v0,1).*dx);
    
    xNeg=x<=0;

    vtMagnus=v0;
    A=-sqrt(rho).*Dx;
    B=Dxx./2-beta.*Dx;
    for Ti=1:1:length(Tj)-1
        currMt=reshape(Mt(1+(Ti-1)*nEM:1+Ti*nEM,:)-Mt(1+(Ti-1)*nEM,:),1,1,[],M);
        vtMagnus=magnusConst(A,B,vtMagnus,currMt,Tj(Ti+1)-Tj(Ti),order);
        vtMagnus(xNeg,:)=0;
        LtSM(Ti+1,:)=LGD.*(1-sum(vtMagnus,1).*dx);
    end

end