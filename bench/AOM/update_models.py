import re

with open('bench/AOM/models.py', 'r') as f:
    content = f.read()

# 1. Add EnhancedAOMPLSRegressor
enhanced_aom_code = """
# =============================================================================
# 0. Enhanced AOM-PLS (Baseline Clone with more transforms)
# =============================================================================
class EnhancedAOMPLSRegressor(AOMPLSRegressor):
    def fit(self, X, y, X_val=None, y_val=None):
        from nirs4all.operators.transforms import StandardNormalVariate as SNV, SavitzkyGolay as SG, Detrend, MultiplicativeScatterCorrection as MSC
        from nirs4all.operators.transforms.orthogonalization import OSC
        
        # Create a rich operator bank
        snv_op = PseudoLinearSNVOperator()
        snv_op.fit(X)
        
        # We can't easily add non-linear operators like MSC or SG directly to AOM-PLS 
        # unless they are LinearOperators. SG and Detrend are linear!
        # Let's use the default bank + SNV
        self.operators_ = default_operator_bank() + [snv_op]
        
        original_bank = default_operator_bank
        import nirs4all.operators.models.sklearn.aom_pls as aom_module
        aom_module.default_operator_bank = lambda: self.operators_
        try:
            super().fit(X, y, X_val, y_val)
        finally:
            aom_module.default_operator_bank = original_bank
        return self
"""

# 2. Update BanditAOMPLSRegressor
bandit_code = """
# =============================================================================
# 3. Multi-Armed Bandit Operator Selection (Dynamic NIPALS)
# =============================================================================
class BanditAOMPLSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=15):
        self.n_components = n_components

    def fit(self, X, y):
        self.x_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y, axis=0)
        X_c = X - self.x_mean_
        y_c = y - self.y_mean_
        if y_c.ndim == 1:
            y_c = y_c.reshape(-1, 1)
            
        snv_op = PseudoLinearSNVOperator()
        snv_op.fit(X)
        self.operators_ = default_operator_bank() + [snv_op]
        
        for op in self.operators_:
            op.initialize(X.shape[1])
            
        n, p = X_c.shape
        q = y_c.shape[1]
        B = len(self.operators_)
        eps = 1e-12
        
        W = np.zeros((p, self.n_components), dtype=np.float64)
        T = np.zeros((n, self.n_components), dtype=np.float64)
        P = np.zeros((p, self.n_components), dtype=np.float64)
        Q = np.zeros((q, self.n_components), dtype=np.float64)
        
        X_res = X_c.copy()
        Y_res = y_c.copy()
        n_extracted = 0
        
        active_ops = list(range(B))
        self.selected_ops_ = []
        
        for k in range(self.n_components):
            c_k = X_res.T @ Y_res
            if q == 1:
                c_k = c_k[:, 0]
            else:
                u, s, _ = np.linalg.svd(c_k, full_matrices=False)
                c_k = u[:, 0] * s[0]
                
            c_norm = np.linalg.norm(c_k)
            if c_norm < eps:
                break
                
            best_r2 = -np.inf
            best_w_k = None
            best_t_k = None
            best_p_k = None
            best_q_k = None
            best_op_idx = None
            
            op_r2s = []
            
            for b in active_ops:
                op = self.operators_[b]
                g = op.apply_adjoint(c_k)
                g_norm = np.linalg.norm(g)
                if g_norm < eps:
                    op_r2s.append(-np.inf)
                    continue
                w_hat = g / g_norm
                a_w = op.apply(w_hat.reshape(1, -1)).ravel()
                a_w_norm = np.linalg.norm(a_w)
                if a_w_norm < eps:
                    op_r2s.append(-np.inf)
                    continue
                w_k_b = a_w / a_w_norm
                
                t_k_b = X_res @ w_k_b
                tt_b = t_k_b @ t_k_b
                if tt_b < eps:
                    op_r2s.append(-np.inf)
                    continue
                    
                p_k_b = (X_res.T @ t_k_b) / tt_b
                q_k_b = (Y_res.T @ t_k_b) / tt_b
                
                r2 = tt_b * np.sum(q_k_b**2)
                op_r2s.append(r2)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_w_k = w_k_b
                    best_t_k = t_k_b
                    best_p_k = p_k_b
                    best_q_k = q_k_b
                    best_op_idx = b
                    
            if best_w_k is None:
                break
                
            # Prune operators that are significantly worse than the best
            new_active_ops = []
            for b, r2 in zip(active_ops, op_r2s):
                if r2 >= 0.5 * best_r2:
                    new_active_ops.append(b)
            active_ops = new_active_ops
            self.selected_ops_.append(self.operators_[best_op_idx].name)
            
            W[:, k] = best_w_k
            T[:, k] = best_t_k
            P[:, k] = best_p_k
            Q[:, k] = best_q_k
            
            X_res = X_res - np.outer(best_t_k, best_p_k)
            Y_res = Y_res - np.outer(best_t_k, best_q_k)
            n_extracted = k + 1
            
        self.n_components_ = n_extracted
        self.B_coefs_ = np.zeros((p, q), dtype=np.float64)
        if n_extracted > 0:
            W_ext = W[:, :n_extracted]
            P_ext = P[:, :n_extracted]
            Q_ext = Q[:, :n_extracted]
            PTW = P_ext.T @ W_ext
            PTW.flat[::n_extracted + 1] += 1e-12
            inv_PTW = np.linalg.inv(PTW)
            self.B_coefs_ = W_ext @ inv_PTW @ Q_ext.T
            
        return self

    def predict(self, X):
        X_c = X - self.x_mean_
        return (X_c @ self.B_coefs_ + self.y_mean_).flatten()
"""

# Replace BanditAOMPLSRegressor
content = re.sub(r'# =============================================================================\n# 3\. Multi-Armed Bandit Operator Selection\n# =============================================================================\nclass BanditAOMPLSRegressor.*?return self', bandit_code.strip(), content, flags=re.DOTALL)

# Insert EnhancedAOMPLSRegressor before PseudoLinearSNVOperator
content = content.replace('# =============================================================================\n# 1. Pseudo-Linearization', enhanced_aom_code.strip() + '\n\n# =============================================================================\n# 1. Pseudo-Linearization')

with open('bench/AOM/models.py', 'w') as f:
    f.write(content)
