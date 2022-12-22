function fig=plotInitialDatum(x0,v0,x,figName,varargin)
    if nargin<4
        figName='initialDatum.pdf';
    end
    fig=newFigure(varargin); hold on;
    
    histogram(x0,'Normalization','pdf');    
    plot(x,v0,'b-');
    
    iLeft = max(find(v0>0,1,'first')-5,1);
    iRight = min(find(v0>0,1,'last')+5,length(x0));
    
    
    xlim([x(iLeft),x(iRight)]);

    legend({'$x_0$','$v^{(0)}(x)$'},'Interpreter','latex','NumColumns',2,'Orientation','horizontal','Location','southoutside');
    exportgraphics(fig,figName)
end