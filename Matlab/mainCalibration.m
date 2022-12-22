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

%% Load data
disp('Load data')
iTRAXX_Date = '26_09_22'; 
r=.015; % interest rate for 26_09_22
% iTRAXX_Date = '05_12_22'; 
% r=.026; % interest rate for 05_12_22

iTRAXX_Path = '../Data/iTRAXX';
[iTRAXX_Tranches, iTRAXX_Tranches_Values, iTRAXX_Tranches_Year] =...
    load_iTRAXX_Tranches(iTRAXX_Path,iTRAXX_Date);
index = iTRAXX_Tranches_Values(1)/10000;
spreads = iTRAXX_Tranches_Values(2:end);
spreads(1:2)=spreads(1:2)*100; % first two tranches in percent
spreads=(spreads+100)./10000; % upfront adjustment and convert to bps
aPoints = reshape(iTRAXX_Tranches(:,1),1,1,[]);
dPoints = reshape(iTRAXX_Tranches(:,2),1,1,[]);
%% Load Trained Neural Networks
disp('Load Neural Network')
sRange=[.01,.5];
rhoRange=[.01,.99];
x0Range=[0,6];

modelIni = ['..\Model4Matlab\IniNet\r',num2str(r),...
            '_sigma(',num2str(sRange(1)),', ',num2str(sRange(2)),...
            ')_rho(',num2str(rhoRange(1)),', ',num2str(rhoRange(2)),...
            ')_x0(',num2str(x0Range(1)),', ',num2str(x0Range(2)),...
            ')_BM100000_T5.0_N21_',iTRAXX_Date,'\IniNet'];
IniNet = importTensorFlowNetwork(modelIni,TargetNetwork="dlnetwork");
%% Parameters
% finite time horizon
T=iTRAXX_Tranches_Year; % in years
annuity=.25; % quarterly

% time points in one interval [T_{i-1},T_i]
nDM=1;  % Determ Magnus time points 

% number of simulations
M=10^4;

% loss given default
LGD=.6;

% space grid
d=201;  % space discretization
xmin=-10;
xmax=20;
x=linspace(xmin,xmax,d+2);
dx=(xmax-xmin)./(d+1);


%% Time steps
J=(1/annuity)*T+1; % time horizon
Tj=0:annuity:T; % resettlement dates and today

% Determ Magnus time steps
Ndm=(J-1)*nDM+1; 
dtDM=T/(Ndm-1); 


%% Brownian motions
disp('Simulate Brownian motions')
ticBM=tic;
[dMvec,Mvec,tIndvec]=BMcommon(T,[Ndm],M);
ctimeBM=toc(ticBM);
fprintf('Elapsed time for BMs %g s.\n',ctimeBM)
dMt = dMvec{1};

%% Calibration
ticCal=tic;
[err,p]=calibrateDM(r,sRange,rhoRange,IniNet,index,spreads(1:end-1),aPoints(1,1,1:end-1),dPoints(1,1,1:end-1),Tj,x,dMt,LGD,annuity);
ctimeCal = toc(ticCal);
fprintf('Elapsed time for calibration %g s.\n',ctimeCal)
% y=fminDM(p,r,IniNet,index,spreads,aPoints,dPoints,Tj,x,dMt,LGD,annuity);
%%
sigma=p(1);
rho=p(2);
beta=(r-sigma.^2./2)./sigma;
x0 = double(abs(extractdata(predict(IniNet,dlarray(reshape([beta,rho],[],2,1),'BTC')))));
v0 = initialDatum(x0,x(2:end-1));
LtDM=portfolioLossDM([],Tj,x,dMt,beta,rho,v0,LGD);
cDM=STCDOspread(aPoints(1:end-1),dPoints(1:end-1),r,LtDM,Tj);
iDM=STCDOindex(LGD,r,LtDM,Tj,annuity);
%% Save Results
save(['../Results/Mat/',iTRAXX_Date,'.mat'],'p','r','x0','cDM','iDM','spreads','index','ctimeCal');
figInitialDatum=plotInitialDatum(x0,v0,x(2:end-1),['../Results/Figures/initialDatum_',iTRAXX_Date,'.pdf']);

varNames=squeeze(arrayfun(@(a,d) ['[',num2str(a),',',num2str(d),']'],aPoints(1,1,1:end-1),dPoints(1,1,1:end-1),'UniformOutput',false));
varNames(end+1)={'Index'};
errors= 100.*abs([cDM',iDM] - [spreads(1:end-1)',index])./([spreads(1:end-1)',index]);
data=[[cDM',iDM].*10000;[spreads(1:end-1)',index].*10000;errors];
spreadTable=array2table(data, ...
    'VariableNames',varNames,...
    'RowNames',{'Market','Calibration','Error (in %)'});
disp(spreadTable);
table2latex(spreadTable,['../Results/Tex/calError',iTRAXX_Date,'.tex'])


%% Save results
compileLatex=false;
% to do output

%% Clean up
if availableGPUs > 0
    clearGPU=parfevalOnAll(@gpuDevice,0,[]);
    wait(clearGPU)
end
disp('done')