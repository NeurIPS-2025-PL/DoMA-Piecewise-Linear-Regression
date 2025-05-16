function f = dmax_fit(y,x_sample,K)
for ii= 1:1
    M = 2e3; max_iter = 20e3; mu = 1;
    [d,n] = size(x_sample); d=d-1;
    A_per=initialization(y,x_sample,K,M,n,d,ii);
    [est_beta,isstopC] = GD_Dmaxaffine(A_per,mu,max_iter,x_sample,y',n,K,d);
    f = @(x_sample) DMaxAffine_func(est_beta,[x_sample.';ones(1,size(x_sample,1))],K).';
    if isstopC
        break

    else
        disp("repeating till convergence")
    end
end
end