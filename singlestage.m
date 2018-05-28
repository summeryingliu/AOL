%we want to run a simulation for AMOL with OWL for single stage
function V=singlestage(n,seed,p,randomizationconst,sim)
addpath '/hmt/sirius1/skv0/u/3/y/yl2802/common'
V=zeros(sim,4);
m=20000;
rng(135);
mu=zeros(p,1);
sigma=eye(p);
sigma(1:10,1:10)=0.2*ones(10,10)+0.8*eye(10);
Ht = mvnrnd(mu,sigma,m);
e=1e-5;
clinear=2.^(-15:15);
for i=1:sim
    rng((seed-1)*sim+i);
    A = 2*(rand(n,1)>1/2)-1;
    H = mvnrnd(mu,sigma,n);
    R = scesingle(A,H)+randn(n,1);
    if (randomizationconst==1)
        pi = 0.5*ones(n,1);
    else pi = 1./(1+exp(-0.5.*H(:,1)));
    end
    %Single stage OWL as in Zhao's paper
    model1=OLearning_Single(A,H,R,pi,clinear,4,e);
    At=Olearning_predict_Single(Ht,model1);
    inter1 = bsxfun(@times,Ht,At);
    Rt=scesingle(At,Ht);
    V(i,1)=mean(Rt);  
    
    %take residual and shift by -min(r)
    model2=OLearning_Singleonlyres(A,H,R,pi,clinear,4,e);
    At=Olearning_predict_Single(Ht,model2);
    inter1 = bsxfun(@times,Ht,At);
    Rt=scesingle(At,Ht);
    V(i,2)=mean(Rt);
       
    %doesnot take the residual but doesnot shift the R's
    model3=OLearning_Singleabs(A,H,R,pi,clinear,4,e);
    At=Olearning_predict_Single(Ht,model3);
    inter1 = bsxfun(@times,Ht,At);
    Rt=scesingle(At,Ht);
    V(i,3)=mean(Rt);
    
    %Proposed adjusted Olearning
    model4=OLearning_Singler(H,A,R,pi,clinear,4,e);
    At=Olearning_predict_Single(Ht,model4);
    inter1 = bsxfun(@times,Ht,At);
    Rt=scesingle(At,Ht);
    V(i,4)=mean(Rt);         
end

