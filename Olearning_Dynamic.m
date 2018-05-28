function models=Olearning_Dynamic(AA,HH,RR,pi,K,n,clinear,m,e)
if nargin < 9
    e=1e-8;
end

if nargin < 8
    m=10;
end 

if nargin <7
    clinear=2.^(-15:15);
end 
select = true(n,1);
R_future=0;
prob=ones(n,1);
models=cell(K,1);
for j = K:-1:1
    H=HH{j};
    A=AA{j};
    R=(RR{j}+R_future);
    prob=prob.*pi{j};
    models{j}=OLearning_Single(A(select),H(select,:),R(select),prob(select),clinear,m,e);
    tempind=zeros(n,1);
    tempind(select==1)=models{j}{4};
    select=select&tempind;
    R_future=R_future+RR{j};
    
end
