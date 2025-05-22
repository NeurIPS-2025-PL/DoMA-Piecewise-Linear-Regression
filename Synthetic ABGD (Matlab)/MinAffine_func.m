function [y,j]=MinAffine_func(A,x) 
    %MaxAffine funxtion A-> (d+1) x k, x-> (d+1) x n
    tmp=A'*x;
    if size(tmp,1)>1
        [y,j]=min(tmp);
    else
        y=tmp;
        j=ones(1,size(x,2));
    end
end