function y=fminDM(p,r,IniNet,index,spreads,aPoints,dPoints,Tj,x,dMt,LGD,annuity)
    sigma = p(1);
    rho = p(2);
    beta=(r-sigma.^2./2)./sigma;

    x0 = double(abs(extractdata(predict(IniNet,dlarray(reshape([beta,rho],[],2,1),'BTC')))));
    v0 = initialDatum(x0,x(2:end-1));

    LtDM=portfolioLossDM([],Tj,x,dMt,beta,rho,v0,LGD);
    cDM=STCDOspread(aPoints,dPoints,r,LtDM,Tj);  
    iDM=STCDOindex(LGD,r,LtDM,Tj,annuity);

    y1 = cDM - spreads;
    y2 = iDM - index;

    y=[y1(:);y2(:)];
end