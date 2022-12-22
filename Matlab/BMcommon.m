function [dWvec,Wvec,tIndvec]=BMcommon(T,Nvec,M)
%%BROWNIANMOTION computes the Brownian motion with different time steps for
% Euler and Magnus
%   Input:
%       T (1 x 1 double): 
%       	the finite time horizon
%       Nvec (n x 1 array): 
%           the number of time-steps 
%       M (1 x 1 int): 
%           the number of simulations
%   Output:
%       dWvec (n x 1 cell with 1 x 1 x Nvec(i) x M entries): 
%           increments of BMs 
%       tIndvec (n x 1 cell): 
%           indices to match time grids

    [Nvec,sortInd] = sort(Nvec,2,"descend");
    Nmax = Nvec(1);
    tIndvec=cell(length(Nvec)-1,1);
    dWvec=cell(length(Nvec),1);
    Wvec=cell(length(Nvec),1);

    dt = T/(Nmax(1)-1);

    dW =sqrt(dt).*randn(Nmax-1,M);
    dWvec{1}=dW;

    W=zeros(Nmax,M);
    W(2:end,:)=dW;
    W=cumsum(W,1);
    Wvec{1}=W;
    
    for i=2:length(Nvec)
        N=Nvec(i);
        [dWcoarse,Wcoarse,tInd]=coarseBM(N);
        dWvec{i}=dWcoarse;
        Wvec{i}=Wcoarse;
        tIndvec{i-1}=tInd;
    end
    dWvec=dWvec(sortInd);
    iMax=find(Nmax==Nvec,1,"first");
    tIndvec=tIndvec(sortInd([1:iMax-1,iMax+1:length(sortInd)])-1);

    function [dWcoarse,Wcoarse,tInd]=coarseBM(N)
        tInd=1:1:N;
        tInd(2:1:end)=tInd(1:1:end-1).*floor((Nmax-1)/(N-1))+1;
        if mod((Nmax-1),(N-1))~=0
            error('Sub time grids incompatible')
        end
        Wcoarse=W(tInd,:);
        dWcoarse=diff(Wcoarse,1,1);
    end
end
