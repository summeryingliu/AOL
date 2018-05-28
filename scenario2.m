%test 4 stage P-learning
function V=scenario2(n,seed,p,randomizationconst,sim)
V=zeros(sim,3);
Rt=cell(4,1);
At=cell(4,1);
Ht=cell(4,1);
K=4;
m=20000;
rng(135);
mu=zeros(p,1);
sigma=eye(p);
if(p==6)
  sigma(1:3,1:3)=[1,0.3,0.3;0.3,1,0.3;0.3,0.3,1];
else 
  sigma(1:10,1:10)=0.2*ones(10,10)+0.8*eye(10);
end
Ht{1} = mvnrnd(mu,sigma,m);
e=1e-5;
clinear=2.^(-15:15);
R=cell(4,1);
A=cell(4,1);
pi=cell(4,1);
H=cell(4,1);
for j=1:sim
    rng((seed-1)*sim+j);
    for i=1:K
        A{i} = 2*(rand(n,1)>1/2)-1;
    end    
    H{1} = mvnrnd(mu,sigma,n);
    for i=1:(K-1)
       R{i} = sce2(A,H,R,i)+randn(n,1);
       inter = bsxfun(@times,H{i},A{i});
       H{i+1} = [H{i} A{i} inter R{i}];
    end
    R{4} = sce2(A,H,R,4)+randn(n,1);
if (randomizationconst==0)
    pi{1} = 0.5*ones(n,1);
    pi{2} = 0.5*ones(n,1);
    pi{3} = 0.5*ones(n,1);
    pi{4} = 0.5*ones(n,1);
else
    pi{1} = 1./(1+exp(-0.5.*H{1}(:,1)));
    pi{2} = 1./(1+exp(0.1.*R{1}));
    pi{3} = 1./(1+exp(0.2.*H{1}(:,3)));
    pi{4} = 1./(1+exp(0.2.*H{1}(:,4)));
end
    %optimal P-learning
    model1=Plearningbestlasso(A,H,R,pi,K,n,clinear,4,e);
    for i=1:(K-1)
    	At{i}=Olearning_predict_Single(Ht{i},model1{i});
    	inter1 = bsxfun(@times,Ht{i},At{i});
    	Rt{i}=sce2(At,Ht,Rt,i);
    	Ht{i+1} = [Ht{i} At{i} inter1 Rt{i}];
    end
    At{K}=Olearning_predict_Single(Ht{K},model1{K});
    Rt{K}=sce2(At,Ht,Rt,K);
    V(j,1)=mean(Rt{1}+Rt{2}+Rt{3}+Rt{4});   

    %O-learning as in Zhao's paper
    model2=Olearning_Dynamic(A,H,R,pi,K,n,clinear,4,e);
    for i=1:(K-1)
    	At{i}=Olearning_predict_Single(Ht{i},model2{i});
    	inter1 = bsxfun(@times,Ht{i},At{i});
    	Rt{i}=sce2(At,Ht,Rt,i);
    	Ht{i+1} = [Ht{i} At{i} inter1 Rt{i}];
    end
    At{K}=Olearning_predict_Single(Ht{K},model2{K});
    Rt{K}=sce2(At,Ht,Rt,K);
    V(j,2)=mean(Rt{1}+Rt{2}+Rt{3}+Rt{4});

%Qlearning lasso
model3=Qlearning_Dynamiclasso(A,H,R,K);
    for i=1:(K-1)
       At{i}=Qlearning_pred(Ht{i},model3{i});
       if At{i}(1)==0 
         At{i}=2*(rand(m,1)>1/2)-1;
       end
       inter1 = bsxfun(@times,Ht{i},At{i});
       Rt{i}=sce2(At,Ht,Rt,i);
    	Ht{i+1} = [Ht{i} At{i} inter1 Rt{i}];
    end
    At{K}=Qlearning_pred(Ht{K},model3{K});
    if At{K}(1)==0 
         At{i}=2*(rand(m,1)>1/2)-1;
       end
    Rt{K}=sce2(At,Ht,Rt,K);
    V(j,3)=mean(Rt{1}+Rt{2}+Rt{3}+Rt{4});
end
