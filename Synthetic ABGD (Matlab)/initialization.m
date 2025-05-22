function [initial_beta]=initialization(y,x,ns,M,n,p,true_parameter,index_set,s)
       U_hat = eye(min(p,ns));
       [initial_beta,~]=randomsearch(U_hat,x,y,M,ns,p,true_parameter,index_set);
end





function [initial_beta,val]=randomsearch(U,x,y,M,ns,d,A,index_set)
     
        V=[U zeros(d,1); zeros(1,ns), 1];        
        prior_val=inf;
        for m=1:M
            rng(m);
            v_candidate=make_prmtr_v2(ns+1,ns,m*10-1);
            param_tmp=V*v_candidate; %initialization -> A_per 
            val=norm(y-DMaxAffine_func(param_tmp,x,ns));
            if prior_val>val
                prior_val = val;
                candidate=param_tmp;
            end
        end
        
        candiate1=zeros(1,size(index_set,1));  
        for kk=1:size(index_set,1)
                     tmpp=candidate(:,index_set(kk,:));
                     candiate1(kk)=norm(tmpp-A,'fro')^2;
        end
        [val,idx]=min(candiate1);
        initial_beta=candidate(:,index_set(idx,:));
end

function [A]=make_prmtr_v2(d,ns,i)
        %x is one of samples
        rng(100*d+10*ns+i);
        A=rand(d,ns);  %d>ns
        
       
        
end

