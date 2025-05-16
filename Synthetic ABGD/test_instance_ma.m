function test_instance_ma(idx_instance,idx_param,param)

gpu_flag = param.gpu_flag;
cases = param.cases;
covgen = param.covgen;
verbose = 1;
wellbalance=param.wellbalance;
size_n_list = numel(param.n_list);
idx_n = mod(idx_param-1,size_n_list)+1;
n = param.n_list(idx_n);
max_iter = param.maxiter;
mu = param.mu;
n_max = max(param.n_list);
M = param.M;
m = param.m;


switch cases
    
    case 'kvsn'
        
        idx_K = (idx_param-idx_n)/size_n_list+1;
        K = param.K_list(idx_K);
        p = param.p;
        K_max = max(param.K_list);

        
    case 'pvsn'
        
        idx_p = (idx_param-idx_n)/size_n_list+1;
        p = param.p_list(idx_p);
        %p =param.p;
        %idx_s = (idx_param-idx_n)/size_n_list+1;
        %s = param.s_list(idx_s);
        %s = param.s;
        s = p;
        K = param.K;
        
        
    case 'n'
        
        K = param.K;
        p = param.p;
        
        
    case 'minibatch'
        
       idx_m = (idx_param-idx_n)/size_n_list+1;
       m = param.m_list(idx_m);
       K = param.K;
       p = param.p;
    
        
        
    otherwise
        
        error([cases ' is not supported']);
        
        
end


index_set=perms(1:K);
% p = param.p_list(idx_p);

% p_max = max(param.p_list);
sigma = param.sigma;
if gpu_flag==1
    fname = ['./Results/tmp' 'n' num2str(n) 'K' num2str(K) 'p' num2str(p) 'sigma' num2str(sigma) cases covgen 'gpu' '.mat'];
else
    fname = ['./Results/tmp' 'n' num2str(n) 'K' num2str(K) 'p' num2str(p) 'sigma' num2str(sigma) cases covgen '.mat'];
end


if exist(fname,'file')
    disp([fname ' already done']); 
    return;
    
else

    % generate groundtruth X
    
    
    switch cases
        
        case 'kvsn'
        
            rng_seed_X = mod(idx_instance*(p),2^32);
            rng(rng_seed_X);  

             % generate ground_truth
            tmpmat=randn(p+1,K_max);
            if wellbalance == 1
                 [U, ~, V]=svd(tmpmat(1:p,1:K));
                 Theta=U(:,1:K)*eye(K)*V';
                 A=[Theta; zeros(1,K)];        
            else
                A=zeros(p+1,K);
                for j=1:K
                    A(:,j)= tmpmat(:,j)/norm(tmpmat(:,j));
                end
            end
        
        case 'pvsn'
        
            rng_seed_X = mod(idx_instance*(K),2^32);
            rng(rng_seed_X);  
             tmpmat=randn(s+1,K);
            if wellbalance == 1
                 [U, ~, V]=svd(tmpmat(1:s+1,1:K));
                 Theta=U(:,1:K)*eye(K)*V';
                 A=[zeros(p-s,K);Theta];
                 A(1:p,:) = A(randperm(p),:);
            else
                A=zeros(p+1,K);
                for j=1:K
                    A(:,j)= tmpmat(:,j)/norm(tmpmat(:,j));
                end
            end
  
        case 'n'
            
            rng_seed_X = mod(idx_instance*(K+p),2^32);
            rng(rng_seed_X);  

             % generate ground_truth
             
            %% Ground Truth -> Give sparsity (well-balance or randomly generated)
            tmpmat=randn(p+1,K);
            if wellbalance == 1
                 [U, ~, V]=svd(tmpmat(1:p,1:K));
                 Theta=U(:,1:K)*eye(K)*V';
                 A=[Theta; zeros(1,K)];        
            else
                A=zeros(p+1,K);
                for j=1:K
                    A(:,j)= tmpmat(:,j)/norm(tmpmat(:,j));
                end
            end
            
            
         case 'minibatch'
            
            rng_seed_X = mod(idx_instance*(K+p),2^32);
            rng(rng_seed_X);  

             % generate ground_truth
            tmpmat=randn(p+1,K);
            if wellbalance == 1
                 [U, ~, V]=svd(tmpmat(1:p,1:K));
                 Theta=U(:,1:K)*eye(K)*V';
                 A=[Theta; zeros(1,K)];        
            else
                A=zeros(p+1,K);
                for j=1:K
                    A(:,j)= tmpmat(:,j)/norm(tmpmat(:,j));
                end
            end
            
            
        otherwise
            
         error([cases ' is not supported']);
    end
    
    
   
    clear tmpmat;

    % generate measurement model
    rng_seed_A = mod(idx_instance,2^32);
    rng(rng_seed_A);

    switch covgen
        
        case 'gaussian'
        
        tmpmat = randn(p,n_max);
        x_sample = [tmpmat(:,1:n); ones(1,n)];
        
        case 'uniform'
        
        lower_bound = -sqrt(3);
        upper_bound = sqrt(3);
        tmpmat = lower_bound + (upper_bound - lower_bound) * rand(p, n_max);
        x_sample=[tmpmat(:,1:n); ones(1,n)];  

        otherwise
            
         error([covgen ' is not supported']);
    end
    clear tmpmat;

    % generate noisy measurements
    rng_seed_noise = mod(idx_instance,2^32);
    rng(rng_seed_noise);

    tmpmat = randn(1,n_max);
    [y, ~,~]= DMaxAffine_func(A,x_sample,K);
    y = y + sigma*tmpmat(1,1:n);
    
    clear tmpmat;

    tmpres = [];

    %initialization
    A_per=initialization(y,x_sample,K,M,n,p,A,index_set,s);
    %nstd = norm(A);
    %A_per = A + nstd/10*randn(size(A));
    if gpu_flag==1
            x_sample=gpuArray(x_sample);
            y=gpuArray(y);
            A_per=gpuArray(A_per);
    end

    
    
    for idx_alg = 1:numel(param.algs)
        alg = param.algs{idx_alg};

        tic

        switch alg

            case 'am'

                % estimate 
                [est_beta, ~]=AMalgorithm_affine(x_sample,y',K,A_per,max_iter);            

               
            case 'gd'

                est_beta = GD_Dmaxaffine(A_per,mu,max_iter,x_sample,y',n,K,s,A); 
                diffbias = A - est_beta;
                minbias = 1/K *sum(diffbias,2);
                est_beta = est_beta+ minbias;
            case 'pgd'
                
                est_beta = perturbed_GD_maxaffine(A_per,mu,max_iter,x_sample,y',n,K,0.5);

            case 'sgd'
                
                est_beta = SGD_maxaffine(A_per,mu*min(1,m/p),max_iter,x_sample,y',n,m,K);
               

            otherwise
                error([alg ' is not supported']);

        end

        t = toc;

        
       % error estimation    
       sum_tmp=zeros(1,size(index_set,1));          
       for kk=1:size(index_set,1)
           sum_tmp(kk)=norm(est_beta(:,index_set(kk,:))-A,'fro')^2;
       end
       tmpres.error_2norm(idx_alg) = min(sum_tmp)/norm(A,'fro')^2;
                 
        
       
       log10(tmpres.error_2norm(idx_alg))
       % runtime
       tmpres.runtime(idx_alg) = t;
        
       
    end
       % save temporary file
    save(fname,'tmpres');
        
    if verbose == 1
            disp([fname ' instance' num2str(idx_instance) ' finished in ' num2str(t) ' sec']);
    end

    
end

end