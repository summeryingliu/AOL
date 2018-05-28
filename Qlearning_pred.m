function r=Qlearning_pred(X, co)
n=size(X,1);
XX1=[ones(n,1) X ones(n,1) eye(n)*X];
XX2=[ones(n,1) X -ones(n,1) -eye(n)*X];
Q1=XX1*co;
Q2=XX2*co;
r=sign(Q1-Q2);

