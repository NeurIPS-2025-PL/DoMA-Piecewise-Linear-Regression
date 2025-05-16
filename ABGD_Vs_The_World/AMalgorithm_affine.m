function [A_per, iter]=AMalgorithm_affine(x_sample,y,K,A_per,max_iter)
    for iter=1:max_iter
        prior_beta_hat_matrix=A_per;
        [~,dt]=MaxAffine_func(A_per,x_sample);     
        for j=1:K
            A=x_sample(:,(dt==j))'; % indexing
            y_tmp=y(dt==j);
            A_per(:,j)= pinv(A)*y_tmp;
        end
        
        
         if (norm(prior_beta_hat_matrix-A_per,'fro')/norm(prior_beta_hat_matrix,'fro'))<10^(-8)
            break; 
         end      
    end    
end

