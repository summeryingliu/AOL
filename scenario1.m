%testing for Plearningbest for a two stage scenario
function V=scenario1(n,seed,p,randomizationconst,sim)
%addpath '/hmt/sirius1/skv0/u/3/y/yl2802/common'
V=zeros(sim,3);
Rt=cell(2,1);
At=cell(2,1);
Ht=cell(2,1);
K=2;
m=20000;
rng(135);
mu=zeros(p,1);
if(p==3)
  sigma=[1,0.5,0;0.5,1,0;0,0,1];
else 
sigma=eye(p);
sigma(1:10,1:10)=0.2*ones(10,10)+0.8*eye(10);
end
%sigma=ones(p,p);
Ht{1} = mvnrnd(mu,sigma,m);
e=1e-5;
clinear=2.^(-15:15);
R=cell(2,1);
A=cell(2,1);
pi=cell(2,1);
H=cell(2,1);
for i=1:sim
    rng((seed-1)*sim+i);
    A{1} = 2*(rand(n,1)>1/2)-1;
    A{2} = 2*(rand(n,1)>1/2)-1;
    H{1} = mvnrnd(mu,sigma,n);
    R{1} = sce1(A,H,R,1)+randn(n,1);
    inter = bsxfun(@times,H{1},A{1});
    H{2} = [H{1} A{1} inter R{1}];
    R{2} = sce1(A,H,R,2)+randn(n,1);
    if (randomizationconst==1)
        pi{1} = 0.5*ones(n,1);
        pi{2} = 0.5*ones(n,1);
    else pi{1} = 1./(1+exp(-0.5.*H{1}(:,1)));
         pi{2} = 1./(1+exp(0.1.*R{1}));
    end
    %O-learning as in Zhao's paper
    model1=Olearning_Dynamic(A,H,R,pi,K,n,clinear,4,e);
    At{1}=Olearning_predict_Single(Ht{1},model1{1});
    inter1 = bsxfun(@times,Ht{1},At{1});
    Rt{1}=sce1(At,Ht,Rt,1);
    Ht{2} = [Ht{1} At{1} inter1 Rt{1}];
    At{2}=Olearning_predict_Single(Ht{2},model1{2});
    Rt{2}=sce1(At,Ht,Rt,2);
    V(i,2)=mean(Rt{1}+Rt{2});  
    %optimal P learning
    model2=Plearningbestlasso(A,H,R,pi,K,n,clinear,4,e);
    At{1}=Olearning_predict_Single(Ht{1},model2{1});
    inter1 = bsxfun(@times,Ht{1},At{1});
    Rt{1}=sce1(At,Ht,Rt,1);
    Ht{2} = [Ht{1} At{1} inter1 Rt{1}];
    At{2}=Olearning_predict_Single(Ht{2},model2{2});
    Rt{2}=sce1(At,Ht,Rt,2);
    V(i,1)=mean(Rt{1}+Rt{2});   
       

    %Q-learning lasso
    model5=Qlearning_Dynamiclasso(A,H,R,K);
    At{1}=Qlearning_pred(Ht{1},model5{1});
    if At{1}(1)==0 
       At{1}=2*(rand(m,1)>1/2)-1;
    end
    inter1 = bsxfun(@times,Ht{1},At{1});
    Rt{1}=sce1(At,Ht,Rt,1);
    Ht{2} = [Ht{1} At{1} inter1 Rt{1}];
    At{2}=Qlearning_pred(Ht{2},model5{2});
    if At{2}(1)==0 
       At{2}=2*(rand(m,1)>1/2)-1;
    end
    Rt{2}=sce1(At,Ht,Rt,2);
    V(i,3)=mean(Rt{1}+Rt{2});  
end

