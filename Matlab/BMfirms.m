function [Wt,varargout]=BMfirms(T,N,M,K)
%%BROWNIANMOTIONS samples M trajectories of K firm Brownian motions W_t and 
% one common Brownian moton M_t with a homogeneous time grid.
%
%   Input:
%       T (1 x 1 double): finite time horizon
%       N (1 x 1 int): points in time grid for Wt
%       M (1 x 1 int): number of simulations
%       K (1 x 1 int): number of firms
%
%   Output:
%       t (1 x N double): homogeneous time grid
%       Wt (K x N x M double): firm Brownian motions
%       Mt (1 x N x M double): common Brownian motion
%
%   Usage:
%       Wt=brownianMotions(T,N,M,K)
%       [Wt,t]=brownianMotions(T,N,M,K)
%
%   See also:
%       linspace, randn.

    % Check arguments
%     arguments
%         T (1,1) double
%         N (1,1) {isinteger}
%         M (1,1) {isinteger}
%         K (1,1) {isinteger}
%     end

    % time increments
    dt=T/(N-1);

    % Brownian motions
    dWt = sqrt(dt).*randn(K,N-1,M); % BM increment
    Wt = zeros(K,N,M); % temp BM
    Wt(:,2:end,:)=dWt; % BM 
    Wt=cumsum(Wt,2); % BM 

    if nargout>1
        varargout{1}=linspace(0,T,N); % time grid for Wt
    end
end