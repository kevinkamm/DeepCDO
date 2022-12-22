function X=magnusConst(A,B,X0,W,T,order)
%%MAGNUS computes Ito-stochastic Magnus expansion for given matrix
% processes A_t=A.*a(t,Z_t), B.*b(t,Z_t), such that dX_t = B_t X_t dt + A_t X_t dW_t, X_0=X0.
% If X0 is a (d x 1) vector, expmv is used instead of expm for performance
% boost.
%
% Assumptions:
%   - Time grid is homogeneous.
%   - N is the number of time steps for Magnus logarithm and evaluation is
%     only at the terminal time
%   - Logarithm fully vectorized, small size of matrices
%
% Input:
%   A (d x d x 1 x 1 array): 
%       constant matrix 
%   B (d x d x 1 x 1 array): 
%       constant matrix
%   a (1 x 1 x N x M array): 
%       1 dimensional stochastic process
%   b (1 x 1 x N x M array): 
%       1 dimensional stochastic process
%   X0 (empty, d x 1 x 1 x (M), d x d x 1 x (M) array): 
%       initial value, if X0=[] it is assumed to be the identity matrix and
%       last axis is optional for random initial datum
%   dW (1 x 1 x N x M): 
%       increments of the Brownian motion
%   order (1 x 1 int): 
%       order of Magnus expansion
%   
N=size(W,3);
M=size(W,4);
dt=T/(N-1);
timegrid=reshape(linspace(0,T,N),1,1,[],1);

[spExp,deviceExp,deviceLog]=compMode(size(X0,1),size(X0,2),...
                                     issparse(A) && issparse(B));
compCase=sprintf('%d%d',spExp,deviceExp);

if size(X0,2)~=1
    X=zeros(size(A,1),size(A,1),1,M);
else
    X=zeros(size(A,1),1,1,M);
end
switch order
%     case 1
%         parfor i=1:1:M
%             
%         end
    case 2
        A2=A*A;
        BA=comm(B,A);
        IW=lebesgueInt(W,dt,T);
        if deviceLog
            A=gpuArray(A);
            B=gpuArray(B);
            A2=gpuArray(A2);
            BA=gpuArray(BA);
        end
        parfor i=1:M
            Y=secondorder(B,A,A2,BA,T,W(:,:,end,i),IW(:,:,end,i));
            if size(X0,4)>1
                switch compCase %spExp deviceExp
                    case '00'
                        X(:,:,1,i)=Exp(full(gather(Y)),X0(:,:,1,i));
                    case '10'
                        X(:,:,1,i)=Exp(gather(Y),X0(:,:,1,i));
                    case '01'
                        X(:,:,1,i)=Exp(full(Y),gpuArray(single(X0(:,:,1,i))));
                    case '11'
                        X(:,:,1,i)=Exp(Y,gpuArray(X0(:,:,1,i)));
                end
            else
                switch compCase %spExp deviceExp
                    case '00'
                        X(:,:,1,i)=Exp(full(gather(Y)),X0);
                    case '10'
                        X(:,:,1,i)=Exp(gather(Y),X0);
                    case '01'
                        X(:,:,1,i)=Exp(single(full(Y)),gpuArray(single(X0)));
                    case '11'
                        X(:,:,1,i)=Exp(Y,gpuArray(X0));
                end
            end
%             if size(X0,4)>1
%                 switch compCase %spExp deviceExp
%                     case '00'
%                         X(:,:,1,i)=Exp(full(gather(Y)),X0(:,:,1,i));
%                     case '10'
%                         X(:,:,1,i)=Exp(gather(Y),X0(:,:,1,i));
%                     case '01'
%                         X(:,:,1,i)=Exp(full(Y),gpuArray(X0(:,:,1,i)));
%                     case '11'
%                         X(:,:,1,i)=Exp(Y,gpuArray(X0(:,:,1,i)));
%                 end
%             else
%                 switch compCase %spExp deviceExp
%                     case '00'
%                         X(:,:,1,i)=Exp(full(gather(Y)),X0);
%                     case '10'
%                         X(:,:,1,i)=Exp(gather(Y),X0);
%                     case '01'
%                         X(:,:,1,i)=Exp(full(Y),gpuArray(X0));
%                     case '11'
%                         X(:,:,1,i)=Exp(Y,gpuArray(X0));
%                 end
%             end
        end
    case 3
        A2=A*A;
        BA=comm(B,A);
        BAA=comm(BA,A);
        BAB=comm(BA,B);
        IW=lebesgueInt(W,dt,T);
        IW2=lebesgueInt(W.^2,dt,T);
        IsW=lebesgueInt(timegrid.*W,dt,T);
        if deviceLog
            A=gpuArray(A);
            B=gpuArray(B);
            A2=gpuArray(A2);
            BA=gpuArray(BA);
            BAA=gpuArray(BAA);
            BAB=gpuArray(BAB);
        end
        parfor i=1:M
            Y=thirdorder(B,A,A2,BA,BAA,BAB,T,W(:,:,end,i),IW(:,:,end,i),IsW(:,:,end,i),IW2(:,:,end,i));
            if size(X0,4)>1
                switch compCase %spExp deviceExp
                    case '00'
                        X(:,:,1,i)=Exp(full(gather(Y)),X0(:,:,1,i));
                    case '10'
                        X(:,:,1,i)=Exp(gather(Y),X0(:,:,1,i));
                    case '01'
                        X(:,:,1,i)=Exp(single(full(Y)),gpuArray(single(X0(:,:,1,i))));
                    case '11'
                        X(:,:,1,i)=Exp(Y,gpuArray(X0(:,:,1,i)));
                end
            else
                switch compCase %spExp deviceExp
                    case '00'
                        X(:,:,1,i)=Exp(full(gather(Y)),X0);
                    case '10'
                        X(:,:,1,i)=Exp(gather(Y),X0);
                    case '01'
                        X(:,:,1,i)=Exp(single(full(Y)),gpuArray(single(X0)));
                    case '11'
                        X(:,:,1,i)=Exp(Y,gpuArray(X0));
                end
            end
%             if size(X0,4)>1
%                 switch compCase %spExp deviceExp
%                     case '00'
%                         X(:,:,1,i)=Exp(full(gather(Y)),X0(:,:,1,i));
%                     case '10'
%                         X(:,:,1,i)=Exp(gather(Y),X0(:,:,1,i));
%                     case '01'
%                         X(:,:,1,i)=Exp(full(Y),gpuArray(X0(:,:,1,i)));
%                     case '11'
%                         X(:,:,1,i)=Exp(Y,gpuArray(X0(:,:,1,i)));
%                 end
%             else
%                 switch compCase %spExp deviceExp
%                     case '00'
%                         X(:,:,1,i)=Exp(full(gather(Y)),X0);
%                     case '10'
%                         X(:,:,1,i)=Exp(gather(Y),X0);
%                     case '01'
%                         X(:,:,1,i)=Exp(full(Y),gpuArray(X0));
%                     case '11'
%                         X(:,:,1,i)=Exp(Y,gpuArray(X0));
%                 end
%             end
        end
    otherwise
        error('Order %d not implemented',order)
end
end
function I=lebesgueInt(f,dt,T)
    if size(f,3)>1
        I=sum(f(:,:,1:1:end-1,:),3).*dt; 
    else
        I=f.*T; 
    end
end
function C=comm(A,B)
    C=A*B-B*A;
end
function [spExp,deviceExp,deviceLog]=compMode(d1,d2,sp)
%%COMPMODE heuristics for comp device and mode. Can change for different
% hardware. Use code below for timings.
%   Input:
%       d1 (1 x 1 int): Matrix dimension
%       d2 (1 x 1 int): either 1 or d1, decides if expmv or expm is used
%   Output:
%       spExp (1 x 1 boolean): 1 use sparse matrices for Exp, 0 use full
%       deviceExp (1 x 1 boolean): 1 use GPU for Exp, 0 CPU
%       deviceLog (1 x 1 boolean): 1 use GPU for Log, 0 CPU
%   Assumption:
%       expm uses single precision on GPU                           
    if sp
        if d2>1
            if d1<200
                spExp=0;
                deviceExp=0;
                deviceLog=0;
            else
                spExp=1;
                deviceExp=0;%expm cannot use sparse inputs
                deviceLog=1;
            end
        else
            if d1>10000
                spExp=1;
                deviceExp=1;
                deviceLog=1;
            else
                spExp=1;
                deviceExp=0;
                deviceLog=0;
            end
        end
    else
        spExp=0;
        if d1<200
            deviceLog=0;
        else
            deviceLog=1;
        end
        if d1>=400 && d1<=1000
            deviceExp=1;
        else
            deviceExp=0;
        end
    end
    if gpuDeviceCount("available")<1
        deviceLog=0;
        deviceExp=0;
    end
end
function X=Exp(Y,Y0)
d=size(Y,1);
m=30;
precision = 'half';
switch size(Y0,2)
    case 0
        F=expm(Y);
    case 1
        [F,~,~,~]=expmvtay2(Y,Y0,m,precision);
    case d
        F=expm(Y)*Y0;
end
X=gather(F);
end

function O=firstorder(B,A,t,W)
    O=B.*t+A.*W;
end
function O=secondorder(B,A,A2,BA,t,W,IW)
    O=B.*t+A.*W-.5.*A2.*t+BA.*IW-.5.*BA.*t.*W;
end
function O=thirdorder(B,A,A2,BA,BAA,BAB,t,W,IW,IsW,IW2)
    O=B.*t+A.*W+...
          -.5.*A2.*t+BA.*IW-.5.*BA.*t.*W+...
          -(1/12).*BAB.*t.^2.*W+...
          (1/12).*BAA.*t.*W.^2+...
          BAB.*IsW-...
          .5.*BAA.*W.*IW-...
          .5.*BAB.*t.*IW+...
          .5.*BAA.*IW2;
end