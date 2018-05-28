function model=OLearning_Singleabs(A,H,R2,pi,clinear,m,e)
if nargin < 6
    e=1e-10;
end

if nargin < 5
    m=3;
end 

if nargin <4
    clinear=2.^(-15:15);
end 
R2=R2./pi;
npar=length(clinear);
n=length(A);
%rand=mod(randperm(n),m)+1;
rand=mod(1:n,m)+1;
V=zeros(m,npar);
for i = 1:m
        this=(rand~=i);
        X=H(this,:);
        Y=A(this);
        R=R2(this);
        Xt=H(~this,:);
        Yt=A(~this);
        Rt=R2(~this);
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
model=wsvm3(H,A,R2,cbest,e);
