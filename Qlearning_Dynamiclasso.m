function coef=Qlearning_Dynamiclasso(AA,HH,RR,K) 
R_future=0;
coef=cell(K,1);
for j=K:-1:1
    H=HH{j};
    A=AA{j};
    R=RR{j}+R_future;
    if min(R)~=max(R)
        output=Qlearning_Singlelasso(H,A,R);
        coef{j}=output.co;
        R_future=output.Q;
    else
        coef{j}=zeros(2+2*size(H,2),1);
        R_future=R;
    end
end
