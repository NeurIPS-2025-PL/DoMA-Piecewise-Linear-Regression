function f_hat = auto_tune_dmax_fit(y_train,X_train)
% X_train: input covariates n x d
% y_train: input data n x 1
[n,~] = size(X_train);
k_range = 2*(1:4);

n_folds = 5;
loss = nan(1,n_folds);
for jj = 1:numel(k_range)

    loss(jj) = cross_validate(y_train, X_train, ...
        @(y,X) dmax_fit(y_train.', [X_train.';ones(1,n)],k_range(jj)), ...
        n_folds, "regression");
end

[~,i] = min(loss);
k_cv = k_range(i);

fprintf('Optimal K value: %.3f\n\n', k_cv);

f_hat = dmax_fit(y_train.', [X_train.';ones(1,n)],k_cv);

end