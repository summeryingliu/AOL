function out=scesingle(A,H)
    %out=H(:,1).*A-H(:,2).*A+H(:,3);
    out=H(:,1).*A-H(:,2).*A+2*H(:,3)-H(:,4);
end