clear all; close all; fclose('all'); rng(0);
try
    num_workers = str2num(getenv('SLURM_CPUS_PER_TASK'));
    old_threads = maxNumCompThreads(num_workers);
    pool=parpool('threads'); % multithreading
    fprintf('Chosen number of workers %d, Number of active workers %d\n',num_workers,pool.NumWorkers)
catch ME
end
pool=gcp('nocreate');
if isempty(pool)
%     pool=parpool('local'); % multiprocessing
    pool=parpool('threads'); % multithreading
end
availableGPUs = gpuDeviceCount('available');
if availableGPUs > 0
    gpuDevice([]); % clears GPU
    gpuDevice(1); % selects first GPU, change for multiple with spmd
end

%% Figure options
backgroundColor='w';
textColor='k';

%% Parameters
% finite time horizon
T=5; % in years
annuity=.25;

% time points in one interval [T_{i-1},T_i]
nMC=1;  % Monte-Carlo time points 
nEM=15;  % Euler-Maruyama time points 
nTheta=5;  % Theta-scheme time points 
nSM=15;  % Stoch Magnus time points 
nDM=5;  % Determ Magnus time points, legacy dont change 

% number of simulations
M=10^5;
% loss given default
LGD=.6;

% Model parameters

% sigma=0.18; % values from Reisinger2011 p.25 Table 2
% rho=.22; % (=correlation^2) values from Reisinger2011 p.25 Table 2
% r=.042; % initial deterministic interest rate
% x0=normrnd(4.6.*ones(K,1,1),.8.*ones(K,1,1)); % values from Reisinger2011 p.13

iTRAXX_Date = '26_09_22';
% iTRAXX_Date = '05_12_22';
tmp=load(['../Results/Mat/',iTRAXX_Date,'.mat']);
p=tmp.p;
r=tmp.r;
sigma=p(1);
rho=p(2);
x0=tmp.x0;

% size of basket
K=length(x0);

% Large basket parameters
beta=(r-sigma.^2./2)./sigma;

d=201;  % space discretization
xmin=-10;
xmax=20;
x=linspace(xmin,xmax,d+2);
dx=(xmax-xmin)./(d+1);

% theta scheme, 0=explicit, 1=implicit, .5=Crank-Nicolson
theta=0.5;

% order of stochastic Magnus expansion
oSM=2;

%% Time steps
J=(1/annuity)*T+1; % time horizon
Tj=0:annuity:T; % resettlement dates and today

% The largest time factor has to be integer-divisible by the others!

% Monte Carlo time points 
Nmc=(J-1)*nMC+1; 

% Euler-Maruyama time steps
Nem=(J-1)*nEM+1; 
dtEM=T/(Nem-1);

% Theta-scheme time steps
Ntheta=(J-1)*nTheta+1; 
dtTheta=T/(Ntheta-1);

% Stoch Magnus time steps
Nsm=(J-1)*nSM+1; 
dtSM=T/(Nsm-1); 

% Determ Magnus time steps
Ndm=(J-1)*nDM+1; 
dtDM=T/(Nsm-1); 


%% Brownian motions
disp('Simulate Brownian motions')
ticBM=tic;
Wt=BMfirms(T,Nmc,M,K);% Wt shape = (i-th firm, j-th time, m-th path)
[dMvec,Mvec,tIndvec]=BMcommon(T,[Nem,Ntheta,Nsm,Ndm,Nmc],M);
ctimeBM=toc(ticBM);
fprintf('Elapsed time for BMs %g s.\n',ctimeBM)


%% Monte-Carlo simulation
tMC=linspace(0,T,Nmc);
MtMC=reshape(Mvec{end},1,Nmc,M);

ticMC=tic;
[LtMC,tau]=portfolioLossMC(tMC,Tj,Wt,MtMC,r,sigma,rho,x0,LGD);
ctimeMC=toc(ticMC);
fprintf('Elapsed time for portfolio loss with Monte Carlo %g s.\n',ctimeMC)


%% Large basket approximation
% density of initial distance of default projected on x grid
v0=initialDatum(x0,x(2:end-1));

%% Euler-Maruyama
tEM=linspace(0,T,Nem);
MtEM=Mvec{1};

ticEM=tic;
LtEM=portfolioLossEM(tEM,Tj,x,MtEM,beta,rho,v0,LGD);
ctimeEM=toc(ticEM);
fprintf('Elapsed time for portfolio loss with Euler-Maruyama %g s.\n',ctimeEM)


%% Theta-scheme
tTheta=linspace(0,T,Ntheta);
dMt=dMvec{end};

ticTheta=tic;
LtTheta=portfolioLossTheta(tTheta,Tj,x,dMt,beta,rho,v0,LGD,theta);
ctimeTheta=toc(ticTheta);
fprintf('Elapsed time for portfolio loss with Theta-scheme %g s.\n',ctimeTheta)


%% Stochastic Magnus
tSM=linspace(0,T,Nsm);
MtSM=Mvec{1};

ticSM=tic;
LtSM=portfolioLossSM(tSM,Tj,x,MtSM,beta,rho,v0,LGD,oSM);
ctimeSM=toc(ticSM);
fprintf('Elapsed time for portfolio loss with stoch Magnus %g s.\n',ctimeSM)


%% Deterministic Magnus
tDM=linspace(0,T,Ndm);
dMt=dMvec{end};

ticDM=tic;
LtDM=portfolioLossDM(tDM,Tj,x,dMt,beta,rho,v0,LGD);
ctimeDM=toc(ticDM);
fprintf('Elapsed time for portfolio loss with determ Magnus %g s.\n',ctimeDM)


%% STCDO spread 
aPoint=reshape([0,3,6,9,12,22]./100,1,1,[]);
dPoint=reshape([3,6,9,12,22,100]./100,1,1,[]);

cMC=STCDOspread(aPoint,dPoint,r,LtMC,Tj).*10000;  % in bps
cEM=STCDOspread(aPoint,dPoint,r,LtEM,Tj).*10000;  % in bps
cTheta=STCDOspread(aPoint,dPoint,r,LtTheta,Tj).*10000;  % in bps
cSM=STCDOspread(aPoint,dPoint,r,LtSM,Tj).*10000;  % in bps
cDM=STCDOspread(aPoint,dPoint,r,LtDM,Tj).*10000;  % in bps

iMC=STCDOindexMC(LGD,r,1/K,tau,Tj,annuity).*10000;  % in bps
iEM=STCDOindex(LGD,r,LtEM,Tj,annuity).*10000;  % in bps
iTheta=STCDOindex(LGD,r,LtTheta,Tj,annuity).*10000;  % in bps
iSM=STCDOindex(LGD,r,LtSM,Tj,annuity).*10000;  % in bps
iDM=STCDOindex(LGD,r,LtDM,Tj,annuity).*10000;  % in bps

rowNames=arrayfun(@(a,d) ['[',num2str(a),',',num2str(d),']'],aPoint(:),dPoint(:),'UniformOutput',false);
rowNames(end+1)={'Index (in bps)'};
rowNames(end+1)={'ctime (in s)'};
data=[[cMC,cEM,cTheta,cSM,cDM];[iMC,iEM,iTheta,iSM,iDM];[ctimeMC,ctimeEM,ctimeTheta,ctimeSM,ctimeDM]];
spreadTable=array2table(data, ...
    'VariableNames',{'Monte-Carlo','Euler-Maruyama','Theta','Stoch Magnus','Determ Magnus'},...
    'RowNames',rowNames);
disp(spreadTable);


%% Save results
compileLatex=false;
% to do output

%% Clean up
if availableGPUs > 0
    clearGPU=parfevalOnAll(@gpuDevice,0,[]);
    wait(clearGPU)
end
disp('done')