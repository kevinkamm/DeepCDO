function v0=initialDatum(x0,x)
%%INITIALDATUM approximates the dirac initial datum in x0 by the integral 
% of linear splines on a given space grid x.

    d = length(x);
    x0=reshape(x0,1,[]);
    x=reshape(x,[],1);

    dx = (x(end)-x(1))./(d-1);
    phi=min(max(x0-x+dx,0),max(-x0+x+dx,0))./dx;

    v0=mean(phi,2)./dx;
%     sum(v0).*dx % =1
end