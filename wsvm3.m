%weighted SVM, X is feature matrix, Y is treatment, R is weights, c is cost
%variable
function model= wsvm3(X, Y, R, C, e)
if nargin < 4
    C=1;
end 

if nargin < 5
    e=0.00001;
end 

e=e*C;
n=size(Y,1);
YR=Y.*R;
K=X*X';
H=K.*(YR*YR');
%f=-R;
f=-abs(R);
lb=zeros(n,1);
ub=ones(n,1)*C;
Aeq=YR';
beq=0;
%'Display','iter',
%opts=optimset('Algorithm','interior-point-convex','TolX',1e-13,'TolFun',1e-13);
opts=optimset('Algorithm','interior-point-convex','TolX',1e-9,'TolFun',1e-9);
alpha=quadprog(H,f,[],[],Aeq,beq,lb,ub,[],opts)
alpha1=alpha.*R;
w=X'*(alpha1.*Y);
Imid =(alpha < C-e) & (alpha > e);
rm=sign(R).*Y-X*w;
rmid=rm(Imid);
if sum(Imid)>0
    bias=mean(rmid);
else
    Iup=((alpha<e)&(Y==-sign(R)))|((alpha>C-e)&(Y==sign(R)));
    Ilow=((alpha<e)&(Y==sign(R)))|((alpha>C-e)&(Y==-sign(R)));
    rup=rm(Iup);
    rlow=rm(Ilow);
    bias=(min(rup)+max(rlow))/2;
end
% dif=diag(Y)*K*(Y.*alpha1)-ones(n,1);
% Imid =(alpha < C-e) & (alpha > e);
% rm=-Y.*dif;
% rmid = rm(Imid);
% if sum(Imid)>0
%     bias=mean(rmid);
% else
%     Iup=((alpha > C-e) &Y==1)|((alpha<e)&Y==-1);
%     Ilow=((alpha > C-e) &Y==-1)|((alpha<e)&Y==1);
%     rup=-rm(Iup);
%     rlow=-rm(Ilow);
%     Malpha=-max(rup);
%     malpha=-min(rlow);
%     bias=(Malpha+malpha)/2;
% end
% grid=(-20:20)/10;
% nn=size(grid,1);
% hl=zeros(nn,1);
% for j=1:nn
%     pred=grid(j)+X*w;
%     hl(j)=sum(subplus(1-sign(R).*Y.*pred).*abs(R));
% end;
% [~,ind]=min(hl);
% bias=grid(ind);
fit=bias+X*w;
model=cell(1,4);
model{1}=alpha1;
model{2}=bias;
model{3}=w;
model{4}=(sign(fit)==Y);

