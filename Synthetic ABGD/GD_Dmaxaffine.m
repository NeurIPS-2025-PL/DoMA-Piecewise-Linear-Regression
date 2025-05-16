function [A_per]=GD_Dmaxaffine(A_per,mu,max_iter,x_sample,y,n,K,s,A)

verbose = 1;
for i=1:max_iter
    prior_beta_hat_matrix=A_per;
    [y1,dtop,dbot]=DMaxAffine_func(A_per,x_sample,K);
    diff=y1-y';
    tmp = x_sample .* diff;
    matching_indices_top = [double(bsxfun(@eq, dtop(:), 1:K/2)),zeros(n,K/2)];
    matching_indices_bot = [zeros(n,K/2),double(bsxfun(@eq, dbot(:), 1:K/2))];
    for jk = 1:K
        if nnz(matching_indices_top(:,jk)) ~=0
            matching_indices_top(:,jk) = matching_indices_top(:,jk).*n/nnz(matching_indices_top(:,jk));
        end
        if nnz(matching_indices_bot(:,jk)) ~=0
            matching_indices_bot(:,jk) = matching_indices_bot(:,jk).*n/nnz(matching_indices_bot(:,jk));
        end
    end
    if mod(i,2) ~=0
        gradient = tmp * matching_indices_top / n/2;
    else 
        gradient = -tmp * matching_indices_bot / n/2;
    end

    %% Sparsity such as A_per=H_k(A_per-mu*gradient);

    A_per=A_per-mu*gradient;

    if mod(i,1) ==0 || i == max_iter
        for j = 1:K
            [~,I] = maxk(abs(A_per(:,j)),s);
            KeepI = zeros(size(A_per(:,j)));
            KeepI(I) =1;
            A_per(:,j) = A_per(:,j).*KeepI;
        end

    end

    ratio = norm(A_per-prior_beta_hat_matrix,'fro')/norm(prior_beta_hat_matrix,'fro');

    if verbose == 1
        disp(['iteration=' num2str(i) '/' num2str(max_iter) '  stopC=' num2str(ratio)])

        isstopC=ratio<10^(-8);
        if isstopC
            disp('converged')
            break;
        end
    else

        if ratio<10^(-8)
            for j = 1:K
                [~,I] = maxk(abs(A_per(:,j)),s);
                KeepI = zeros(size(A_per(:,j)));
                KeepI(I) =1;
                A_per(:,j) = A_per(:,j).*KeepI;
            end
            break;
        end

    end
end

end