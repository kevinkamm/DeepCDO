function [err,p]=calibrateDM(r,sRange,rhoRange,IniNet,index,spreads,aPoints,dPoints,Tj,x,dMt,LGD,annuity)
    

%     options = optimoptions('lsqnonlin', ...
%                            'Display','iter', ...
%                            'UseParallel',true);
    optLocal = optimoptions('lsqnonlin', ...
                           'Display','iter', ...
                           'UseParallel',false);
    optGA = optimoptions('ga',...
                         'Display','iter', ...
                         'UseParallel',false);
    lb = [sRange(1),rhoRange(1)];
    ub = [sRange(2),rhoRange(2)];

%     p0 = (ub+lb)./2; % does not work
%     p0=ub; % does not work

%     p0=lb; % works well
%     p0=[lb(1) (lb(2)+ub(2))/2]; % works well
    p0=[0.05 0.5]; % works very well

%     p0 = ga(@(p)sum(fminDM(p,r,IniNet,index,spreads,aPoints,dPoints,Tj,x,dMt,LGD,annuity).^2),2,[],[],[],[],lb,ub,[],optGA);
    
    [p,err] = lsqnonlin(@(p)fminDM(p,r,IniNet,index,spreads,aPoints,dPoints,Tj,x,dMt,LGD,annuity),p0,lb,ub,optLocal);
end