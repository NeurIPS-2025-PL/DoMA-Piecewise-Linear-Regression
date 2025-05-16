function [y,jtop,jbot]=DMaxAffine_func(A,x,K) 
    %DMaxAffine funxtion A-> (d+1) x 2k, x-> (p+1) x n
    tmp=A(:,1:K/2)'*x;
    if size(tmp,1)>1
        [ytop,jtop]=max(tmp);
    else
        ytop=tmp;
        jtop=ones(1,size(x,2));
    end
   tmp=A(:,K/2+1:end)'*x;
    if size(tmp,1)>1
        [ybot,jbot]=max(tmp);
    else
        ybot=tmp;
        jbot=ones(1,size(x,2));
    end
y = ytop- ybot;
end
