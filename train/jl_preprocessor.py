# jl_preprocessor.py
import numpy as np
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import warnings

class JohnsonLindenstraussPreprocessor(BaseEstimator, TransformerMixin):
    """
    Johnson-Lindenstrauss preprocessing wrapper for dimensionality reduction.
    
    This transformer reduces feature dimensionality while approximately preserving
    pairwise distances, which is ideal for high-dimensional data with few samples.
    """
    
    def __init__(self, 
                 n_components='auto', 
                 eps=0.1, 
                 sparse=True, 
                 random_state=None):
        """
        Args:
            n_components: Target dimension. If 'auto', computes using JL lemma:
                d >= 4 * log(m) / (eps^2/2 - eps^3/2)
            eps: Maximum distortion tolerance (0 < eps < 1). Smaller = better preservation
            sparse: Use SparseRandomProjection (memory efficient) vs Gaussian
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.eps = eps
        self.sparse = sparse
        self.random_state = random_state
        self.projection_ = None
        self.original_dim_ = None
        self.reduced_dim_ = None
        
    def _compute_auto_dim(self, n_samples):
        """Compute minimum required dimension using JL lemma"""
        # JL lemma lower bound: d >= 4 * log(m) / (eps^2/2 - eps^3/2)
        denominator = (self.eps**2 / 2) - (self.eps**3 / 3)
        if denominator <= 0:
            raise ValueError(f"eps={self.eps} too large, must be < 1.0")
        
        d_min = int(np.ceil(4 * np.log(n_samples) / denominator))
        
        # Ensure at least 1 and not exceeding original dimension
        return max(1, d_min)
    
    def fit(self, X, y=None):
        """Fit the JL projection"""
        X = check_array(X, accept_sparse=False)
        n_samples, n_features = X.shape
        self.original_dim_ = n_features
        
        # Determine target dimension
        if self.n_components == 'auto':
            self.reduced_dim_ = self._compute_auto_dim(n_samples)
            # Don't reduce if already low-dimensional
            if self.reduced_dim_ >= n_features:
                warnings.warn(f"Auto-dimension ({self.reduced_dim_}) >= original dimension ({n_features}). "
                              f"Using original dimension instead.")
                self.reduced_dim_ = n_features
        else:
            self.reduced_dim_ = min(self.n_components, n_features)
        
        # Create projection matrix
        if self.reduced_dim_ < n_features:
            if self.sparse:
                self.projection_ = SparseRandomProjection(
                    n_components=self.reduced_dim_,
                    random_state=self.random_state,
                    eps=self.eps
                )
            else:
                self.projection_ = GaussianRandomProjection(
                    n_components=self.reduced_dim_,
                    random_state=self.random_state,
                    eps=self.eps
                )
            self.projection_.fit(X)
        else:
            # No projection needed
            self.projection_ = None
            
        return self
    
    def transform(self, X):
        """Apply JL projection to reduce dimensionality"""
        X = check_array(X, accept_sparse=False)
        
        if self.projection_ is not None and self.reduced_dim_ < self.original_dim_:
            return self.projection_.transform(X)
        else:
            return X  # No reduction needed
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)
    
    def inverse_transform(self, X_reduced):
        """
        Approximate inverse transform using pseudoinverse.
        Note: This is lossy but can reconstruct approximately.
        """
        if self.projection_ is None:
            return X_reduced
        
        # Get the projection matrix
        if self.sparse:
            P = self.projection_.components_.toarray()
        else:
            P = self.projection_.components_
        
        # Compute pseudoinverse
        P_pinv = np.linalg.pinv(P)
        
        # Reconstruct
        return X_reduced @ P_pinv.T
    
    def get_params(self, deep=True):
        return {
            'n_components': self.n_components,
            'eps': self.eps,
            'sparse': self.sparse,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self