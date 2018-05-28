function model=OLearning_Singlelasso(H,A,R2,pi,clinear,m,e)
if nargin < 6
    e=1e-10;
end

if nargin < 5
    m=3;
end 

if nargin <4
    clinear=2.^(-15:15);
end 

npar=length(clinear);
n=length(A);
p=size(H,2);
%normalize
% mx=mean(H);
% stdX=std(H);
% ONE=ones(n,p);
% Xnorm=(H-ONE*diag(mx))./(ONE*diag(stdX));
% opts=glmnetSet;
if max(R2)~=min(R2)
[fit,fi]=lasso(H,R2,'NumLambda',10,'CV',4);
co=[fi.Intercept(fi.Index1SE);fit(:,fi.Index1SE)];
r=R2-[ones(n,1),H]*co;
else r=R2;
end
%rand=mod(randperm(n),m)+1;
rand=mod(1:n,m)+1;
V=zeros(m,npar);
r=r./pi;
for i = 1:m
        this=(rand~=i);
        X=H(this,:);
        Y=A(this);
        R=r(this);
        Xt=H(~this,:);
        Yt=A(~this);
        Rt=r(~this);
    for j = 1:npar
        c=clinear(j);
        model=wsvm3(X,Y,R,c,e);
        intercept=model{2};
        beta=model{3};
        YP=sign(intercept+Xt*beta);
        V(i,j)=sum(Rt.*(YP==Yt))/length(Yt);
    end
end
mimi=mean(V);
[bc,best]=max(mimi);
cbest=clinear(best);
%bestc=clinear(best);
model=wsvm3(H,A,r,cbest,e);