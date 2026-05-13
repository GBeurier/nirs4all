function model = aom_pls(X, y, varargin)
%AOM_PLS  Fit AOM-PLS (compact, PLS1).
%
%   model = aom_pls(X, y) fits with default options.
%   model = aom_pls(X, y, opts) where opts is a struct with optional fields:
%       max_components   (default 15)
%       n_folds          (default 5)
%       cv_mode          (default 'kfold' ; one of 'kfold','spxy','holdout','external')
%       one_se_rule      (default false)
%       random_state     (default 0)
%       preproc          (default 'none' ; or 'snv','msc','osc','asls','snv+osc','asls+osc')
%       osc_n_components (default 1)
%       asls             struct with lam,p,n_iter
%       center           (default true)
%       external_folds   cell array of int vectors (test indices) — required for cv_mode='external'
%
% Use predict(model, Xnew) to obtain predictions.
%
% Requires the compiled MEX file aompls_mex; see matlab/Makefile or build via:
%     cd matlab && mex -I../cpp/include CXXFLAGS='$CXXFLAGS -std=c++17 -O3' aompls_mex.cpp

if nargin < 2
    error('aompls:usage', 'aom_pls requires X and y');
end
if nargin == 2
    opts = struct();
elseif nargin == 3
    opts = varargin{1};
else
    error('aompls:usage', 'too many arguments');
end
model = aompls_mex('fit', X, y, opts);
end
