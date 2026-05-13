function pred = aom_pls_predict(model, X)
%AOM_PLS_PREDICT  Predict with a fitted aom_pls model.
%   pred = aom_pls_predict(model, X)
pred = aompls_mex('predict', model, X);
end
