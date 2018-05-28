function R=sce2(A,H,R,k)
if k==1
    R=H{1}(:,1).*A{1};
else if k==2
	R=(R{1}+H{1}(:,2).^2+H{1}(:,3).^2-.8).*A{2};
else if k==3
	R=(R{2}+H{1}(:,4)).*A{3}+H{1}(:,5).^2+H{1}(:,6);
else R=(R{3}-0.5).*A{4};
end
end
end