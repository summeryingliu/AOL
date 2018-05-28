function models=Plearningbestlasso(AA,HH,RR,pi,K,n,clinear,m,e)
if nargin < 9
    e=1e-8;
end

if nargin < 8
    m=10;
end 

if nargin <7
    clinear=2.^(-15:15);
end 
select=ones(n,1);
mhat=zeros(n,K);
M=ones(n,K);
C=zeros(n,K);
models=cell(K,1);
prob=ones(n,K);
QLproj=zeros(n,K);
Rsum=0;
R_p=0;
for k=K:-1:1
    H=HH{k};
    A=AA{k}; 
    R=RR{k}+R_p;
    models{k}=OLearning_Singlelasso(H,A,R,pi{k},clinear,m,e);
   %the agreement indicator for stages k-K
    M(:,k:K)=M(:,k:K).*(models{k}{4}*ones(1,K-k+1));
    if (k>1)
        C(:,k:K)=M(:,k-1:K-1)-M(:,k:K);
    end
    if (k==1)
        C(:,2:K)=M(:,1:(K-1))-M(:,2:K);
        C(:,1)=ones(n,1)-M(:,1);
    end
    select=select.*models{k}{4};
    prob(:,k:K)=prob(:,k:K).*(pi{k}*ones(1,K-k+1));
    Rsum=Rsum+RR{k};
    for j=k:K 
        [fit,fi]=lasso(HH{j},Rsum,'NumLambda',10,'CV',4,'Weights',(1-pi{j})./prob(:,j));
        co=fit(:,fi.Index1SE);
        mhat(:,j)=[ones(n,1) HH{j}]*[fi.Intercept(fi.Index1SE);co];
        if (j>1)
            QLproj(:,j)=(C(:,j)-(1-pi{j}).*M(:,j-1))./prob(:,j);
        else
            QLproj(:,1)=(C(:,j)-(1-pi{j}))./prob(:,j);
        end
    end
    %mfunction=Qlearning_Singlelassoweight(H,A,Rsum,(1-pi{k})./prob(:,k));
    %mhat(:,k)=mfunction.Q; 
    R_p=Rsum.*select./prob(:,K)+sum(QLproj(:,k:K).*mhat(:,k:K),2);
end