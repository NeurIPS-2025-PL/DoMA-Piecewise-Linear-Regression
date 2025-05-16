clear all;
clc;

cases_list = {'kvsn','pvsn','n','minibatch'};
covgen_list = {'gaussian', 'uniform'};
cases = cases_list{2};
covgen=covgen_list{1};
gpu_flag=0;
%% parameters
switch cases
    case 'kvsn'   
        K_list = 1:8; % set K_list
        p=50; % set p
        param.K_list = K_list; % set K_list
        param.p = p; % set p
        size_K_list = numel(K_list);
        n_list = 100:100:2500; % T: # block   
    case 'pvsn' 
        p_list = 25;%50:5:400; % set p_list
        K = 2*3; % set K
        s_list = 25;
        param.s_list = s_list;
        param.p_list = p_list;
        %size_s_list = numel(s_list);
        param.p = p_list;
        param.s = s_list;
        param.K = K;
        size_p_list = numel(p_list);
        %n_list =0.4e3:50:4e3; % T: # blocks
        n_list = 1e3;
    case 'n'
        
        p=100;
        K=3;
        param.p = p;
        param.K = K;
        n_list = 2000;
        
        
    case 'minibatch'
        
        p=150;
        K=3;
        n_list=1000:100:3000;
        n_list=1000;
        m_list=25:25:500;
        m_list=50;
        param.m_list=m_list;
        param.K=K;
        param.p=p;
        size_m_list = numel(m_list);
               
    otherwise
        
         error([cases ' is not supported']);
        
end





max_iter = 5E+3;
mu = 1;%0.5; % the constant step size of gradient descent
M = 100; % initialization parameter % How many search iterations
sigma=0; % sigma: the noise deviation. 
wellbalance = 1;


%% setup for Monte Carlo simulation 
m = 100; % mini-batch size
%algs = {'am','gd', 'sgd'};
% algs = {'pgd','sgd'}
algs = {'gd'};


param.m = m;
param.M = M;
param.wellbalance = wellbalance;
param.n_list = n_list;
param.sigma = sigma;
param.algs = algs;
param.cases = cases;
param.covgen = covgen;
param.mu=mu;
param.maxiter=max_iter;
param.gpu_flag=gpu_flag;
size_n_list = numel(n_list);


flag_parallel = 0; % set flag_parallel
total_instance = 5; % 

if flag_parallel == 1
    c = parcluster('local');
    p_for = c.parpool(55); % 8 for macbook air 
end

% run monte carlo 
for idx_instance =1:total_instance
     
    if gpu_flag == 1
        insfname = ['./Results/res_ins' num2str(idx_instance) 'sigma' num2str(sigma) cases covgen 'gpu' '.mat'];
    else
        insfname = ['./Results/res_ins' num2str(idx_instance) 'sigma' num2str(sigma) cases covgen '.mat'];
    end   
    
    if exist(insfname,'file')
        
        disp([insfname ' already done']);
        continue;
        
    else

        TSTART = tic;
        
        if flag_parallel == 1
            
            
                switch cases
                    
                    case 'kvsn'
                        
                         parfor idx_param = 1:size_n_list*size_K_list
                            test_instance_ma(idx_instance,idx_param,param)
                         end
                        
                    case 'pvsn'
                        
                          parfor idx_param = 1:size_n_list*size_p_list                
                            test_instance_ma(idx_instance,idx_param,param)
                          end
                        
                        
                    case 'n'
                                              
                          parfor idx_param = 1:size_n_list               
                                test_instance_ma(idx_instance,idx_param,param)
                          end
                    
                    case 'minibatch'
                        
                           parfor idx_param = 1:size_n_list*size_m_list                
                                test_instance_ma(idx_instance,idx_param,param)
                           end
                        
                        
                     
                    otherwise
                        
                         error([cases ' is not supported']);
                        
                        
                 end
                    
                    
                    
          

            
        else
                       
                
                switch cases
                    
                    case 'kvsn'
                        
                         for idx_param = 1:size_n_list*size_K_list
                            test_instance_ma(idx_instance,idx_param,param)
                         end
                        
                    case 'pvsn'
                        
                         for idx_param = 1:size_n_list*size_p_list 
                         %for idx_param = 1:size_n_list*size_s_list    
                            test_instance_ma(idx_instance,idx_param,param)
                          end
                        
                        
                    case 'n'
                                              
                          for idx_param = 1:size_n_list               
                                test_instance_ma(idx_instance,idx_param,param)
                          end
                    
                    case 'minibatch'
                        
                           for idx_param = 1:size_n_list*size_m_list                
                                test_instance_ma(idx_instance,idx_param,param)
                           end
                        
                        
                     
                    otherwise
                        
                         error([cases ' is not supported']);
                        
                        
                 end
            
        end
            
        ins_t = toc(TSTART);
        
        ins_res = cell(numel(param.algs),1); 
        
        switch cases
            
            case 'kvsn'
                
                
                
                   for idx_param = 1:size_n_list*size_K_list
                        idx_n = mod(idx_param-1,size_n_list)+1;
                        idx_K = (idx_param-idx_n)/size_n_list+1;
                        K = param.K_list(idx_K);
                        n = param.n_list(idx_n);
                        if gpu_flag==1
                            tmpfname = ['./Results/tmp' 'n' num2str(n) 'K' num2str(K) 'p' num2str(p) 'sigma' num2str(sigma) cases covgen 'gpu' '.mat'];  
                        else
                            tmpfname = ['./Results/tmp' 'n' num2str(n) 'K' num2str(K) 'p' num2str(p) 'sigma' num2str(sigma) cases covgen '.mat'];     
                        end
                        load(tmpfname,'tmpres');
                        for idx_alg = 1:numel(param.algs)
                            ins_res{idx_alg}.error_2norm(idx_n,idx_K) = tmpres.error_2norm(idx_alg);
                            ins_res{idx_alg}.error_runtime(idx_n,idx_K) = tmpres.runtime(idx_alg);
                        end
                        delete(tmpfname); 
                   end
                
                
                
                
            case 'pvsn'
                
                
                %for idx_param = 1:size_n_list*size_s_list
                for idx_param = 1:size_n_list*size_p_list
                        idx_n = mod(idx_param-1,size_n_list)+1;
                        idx_p = (idx_param-idx_n)/size_n_list+1;
                        %p = param.p;
                        p = param.p_list(idx_p);
                        n = param.n_list(idx_n);
                        %idx_s = (idx_param-idx_n)/size_n_list+1;
                        %s = param.s_list(idx_s);
                        s = param.s;
                        if gpu_flag==1
                            tmpfname = ['./Results/tmp' 'n' num2str(n) 'K' num2str(K) 'p' num2str(p) 'sigma' num2str(sigma) cases covgen 'gpu' '.mat'];  
                        else
                            tmpfname = ['./Results/tmp' 'n' num2str(n) 'K' num2str(K) 'p' num2str(p) 'sigma' num2str(sigma) cases covgen '.mat'];     
                        end
                        load(tmpfname,'tmpres');
                        for idx_alg = 1:numel(param.algs)
                            if numel(tmpres.error_2norm) < numel(algs)
                                delete(tmpfname);
                                break;
                            end
                            ins_res{idx_alg}.error_2norm(idx_n,idx_p) = tmpres.error_2norm(idx_alg);
                            ins_res{idx_alg}.error_runtime(idx_n,idx_p) = tmpres.runtime(idx_alg);
                        end
                        delete(tmpfname); 
                 end
                
                
                
            case 'n'
                
                
                
                   for idx_param = 1:size_n_list
                        idx_n = mod(idx_param-1,size_n_list)+1;
                        n = param.n_list(idx_n);
                        
                         if gpu_flag==1
                            tmpfname = ['./Results/tmp' 'n' num2str(n) 'K' num2str(K) 'p' num2str(p) 'sigma' num2str(sigma) cases covgen 'gpu' '.mat'];  
                        else
                            tmpfname = ['./Results/tmp' 'n' num2str(n) 'K' num2str(K) 'p' num2str(p) 'sigma' num2str(sigma) cases covgen '.mat'];     
                        end
                        
                        load(tmpfname,'tmpres');
                        for idx_alg = 1:numel(param.algs)
                            ins_res{idx_alg}.error_2norm(idx_n) = tmpres.error_2norm(idx_alg);
                            ins_res{idx_alg}.error_runtime(idx_n) = tmpres.runtime(idx_alg);
                        end
                        delete(tmpfname); 
                   end
                
                
                
                
            case 'minibatch'
                
                
                
                 for idx_param = 1:size_n_list*size_m_list
                        idx_n = mod(idx_param-1,size_n_list)+1;
                        idx_m = (idx_param-idx_n)/size_n_list+1;
                        m = param.m_list(idx_m);
                        n = param.n_list(idx_n);
                        
                        if gpu_flag==1
                            tmpfname = ['./Results/tmp' 'n' num2str(n) 'K' num2str(K) 'p' num2str(p) 'sigma' num2str(sigma) cases covgen 'gpu' '.mat'];  
                        else
                            tmpfname = ['./Results/tmp' 'n' num2str(n) 'K' num2str(K) 'p' num2str(p) 'sigma' num2str(sigma) cases covgen '.mat'];     
                        end
                        load(tmpfname,'tmpres');
                        for idx_alg = 1:numel(param.algs)
                            
                            ins_res{idx_alg}.error_2norm(idx_n,idx_m) = tmpres.error_2norm(idx_alg);
                            ins_res{idx_alg}.error_runtime(idx_n,idx_m) = tmpres.runtime(idx_alg);
                        end
                        delete(tmpfname); 
                  end
                
                
                
                
                
                
                
            otherwise
            
            
            error([cases ' is not supported']);
            
            
            
        end
        
                          
        save(insfname,'ins_t','ins_res'); 

    end
    
end

if flag_parallel == 1
    delete(gcp('nocreate'))
end


% combine all instance results 
res = cell(numel(param.algs),1);
arr_runtime = NaN*ones(total_instance,1); 

for idx_instance = 1:total_instance
    
    if gpu_flag == 1
        insfname =  ['./Results/res_ins' num2str(idx_instance) 'sigma' num2str(sigma) cases covgen 'gpu' '.mat'];
    else
        insfname =  ['./Results/res_ins' num2str(idx_instance) 'sigma' num2str(sigma) cases covgen '.mat'];
    end
     
    if exist(insfname,'file')
        load(insfname,'ins_t','ins_res'); 
        for idx_alg = 1:numel(param.algs)
            res{idx_alg}.error_2norm(:,:,idx_instance) = ins_res{idx_alg}.error_2norm;
            res{idx_alg}.error_runtime(:,:,idx_instance) = ins_res{idx_alg}.error_runtime;
        end
        arr_runtime(idx_instance) = ins_t;
    else
        for idx_alg = 1:numel(param.algs)
            res{idx_alg}.error_2norm(:,:,idx_instance:end) = [];
            res{idx_alg}.error_runtime(:,:,idx_instance:end) = [];
        end
        arr_runtime(idx_instance:end) = [];
        break;
    end
%     delete(insfname); 
end
if gpu_flag==1
    fname = ['./Results/res_' 'sigma' num2str(sigma) cases covgen 'gpu' '.mat'];
else
    fname = ['./Results/res_' 'sigma' num2str(sigma) cases covgen '.mat']; 
end

save(fname,'res','arr_runtime','param');

for idx_alg = 1:numel(param.algs)
log10(median(res{idx_alg}.error_2norm,3))
end


