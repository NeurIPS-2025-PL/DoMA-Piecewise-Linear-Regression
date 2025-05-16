function [initial_beta]=initialization(y,x,ns,M,n,p,ii)
       if p>ns
            [U_hat,~]=PCA_method(y,x,ns,n,p);
       else
            U_hat = eye(p);
       end
       [initial_beta,~]=randomsearch3(U_hat,x,y,M,ns,p,ii);
end



function [U,D_test]=PCA_method(y,x,ns,n,p)
        %x is one of samples
        %ns is the number of segments
%         [d,n]=size(x);
        x=x(1:p,:);
        tmp=1/n*sum(y.*x,2);
         M=tmp*tmp'+1/n*(x*diag(y)*x'-sum(y).*eye(p)); 
%        M=tmp*tmp'+1/n*(x*diag(y)*x'); 
%         sum1=0;
%         for i=1:n
%            sum1=sum1+y(i)*x(:,i)*x(:,i)'-y(i).*eye(p);
%         end
%         1/n*sum1-1/n*(x*diag(y)*x'-sum(y).*eye(p))
        [V,D]=eig(M);
        
        [~,index]=maxk(diag(D),ns);
        U=V(:,index);
        D_test=D(:,index);
end


function [initial_beta,val]=randomsearch3(U,x,y,M,ns,d,ii)
        %x is one of samples
        %ns is the number of segment
%         v = reshape(num2cell(B,1),[ns M]);
     
        V=[U zeros(d,1); zeros(1,min(ns,d)), 1];
        
        
        prior_val=inf;
        for m=1:M
            rng(m);
            
            v_candidate=make_prmtr_v2(min(ns,d)+1,ns,m*10-1,ii);
            
           
       
            param_tmp=AMalgorithm_affine(x,y',ns,V*v_candidate,1); %initialization -> A_per 
            val=norm(y-DMaxAffine_func(param_tmp,x,ns));
            if prior_val>val
                prior_val = val;
                candidate=param_tmp;
            end
        end
        initial_beta = candidate;
        
end

function [A]=make_prmtr_v2(d,ns,i,ii)
        %x is one of samples
        rng(100*d+10*ns+i+ii*10000);
        A=randn(d,ns);  %d>ns
        
       
        
end

