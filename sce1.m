function out=sce1(A,H,R,k)
if k==1
    out=H{1}(:,1).*A{1};
else out=(R{1}+(H{1}(:,2).^2+H{1}(:,3).^2-.8)).*A{2};
end

