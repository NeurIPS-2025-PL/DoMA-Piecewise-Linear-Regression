model_series = zeros(5,5); model_error = zeros(5,5); zn = 1.96; 
for ii = 1:5
    load("ResultsDmax\DataIndex"+string(ii))
    model_series(ii,1)= 100*round(mean(MSE_Dmax),3);
    model_series(ii,2)= 100*round(mean(MSE_DC),3);
    model_series(ii,3)= 100*round(mean(MSE_KNN),3);
    model_series(ii,4)= 100*round(mean(MSE_MARS),3);
    model_series(ii,5)= 100*round(mean(MSE_MLP),3);

    model_error(ii,1)=100*round(zn*std(MSE_Dmax)/sqrt(n_run),3);
    model_error(ii,2)=100*round(zn*std(MSE_DC)/sqrt(n_run),3);
    model_error(ii,3)=100*round(zn*std(MSE_KNN)/sqrt(n_run),3);
    model_error(ii,4)=100*round(zn*std(MSE_MARS)/sqrt(n_run),3);
    model_error(ii,5)=100*round(zn*std(MSE_MLP)/sqrt(n_run),3);
end
b = bar(model_series, 'grouped');
legend(["PL","DC","KNN","MARS","MLP"],Interpreter="latex")
xticklabels(["Acetylene", "Moore","Reaction","Cereal","Carsmall"])
hold on % Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(model_series);
% Get the x coordinate of the bars
x = nan(nbars, ngroups); for i = 1:nbars; x(i,:) = b(i).XEndPoints; end
% Plot the errorbars
errorbar(x',model_series,model_error,'k','linestyle','none'); hold off