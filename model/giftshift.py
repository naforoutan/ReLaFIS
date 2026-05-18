import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import pandas as pd


class GIFTSHIFT(nn.Module):
    """
    GIFT: Gaussian with Integrated Fuzzy Transformation
    Uses sigmoid for "greater than mu" and its negation for "less than mu"
    """
    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool, drop_out_p=0.5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.rules_count = rules
        self.in_features = in_features
        self.out_features = out_features

        self.binary = binary

        if binary:
            self.out_features = out_features = 1

        self.drop_out_p = drop_out_p

        self.device = device

        self.mean = nn.Parameter(torch.rand(
            (in_features, rules), **factory_kwargs))
        self.std = nn.Parameter(torch.rand(
            (in_features, rules), **factory_kwargs))
        self.literal = nn.Parameter(torch.randn(
            (in_features, rules), **factory_kwargs) * 0.1)

        self.local_slopes = nn.Parameter(torch.randn((rules, in_features, out_features), **factory_kwargs) * 0.01)
        self.local_biases = nn.Parameter(torch.zeros((rules, out_features), **factory_kwargs))

        self.decoder_linear = nn.Linear(
            in_features=rules, out_features=in_features, bias=True, **factory_kwargs)

        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(p=drop_out_p)

        self.sigmoid_slope = nn.Parameter(torch.ones(
            (in_features, rules), **factory_kwargs))
        self.temp = nn.Parameter(torch.randn(
            (in_features, rules), **factory_kwargs) * 0.1)
        self.comb_weight = nn.Parameter(torch.randn((in_features, rules), **factory_kwargs) * 0.1) 
        

    def forward(self, X):
        y = self.encode(X)
        entropy = - y * torch.log(y + 1e-10)

        if self.rules_count > 1:
            y = F.normalize(y, p=1, dim=1)

        y = self.drop_out(y)

        reconstructed_X = self.decoder_linear(y)

        y = self.tsk(X, y)

        return y, reconstructed_X, entropy

    def encode(self, X):
        mean = self.mean.view(1, *self.mean.shape)
        std = F.softplus(self.std).view(1, *self.std.shape)
        
        X = X.view(*X.shape, 1)

        def gaussmf(x, mu, sigma):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        # Gaussian membership with negation
        mu_pos = gaussmf(X, mean, std)
        literal = self.sigmoid(self.literal)
        mu_pos_neg = (mu_pos * literal) + (1 - mu_pos) * (1 - literal)

        # Sigmoidal "greater than mu" with negation
        def sigmoidmf(x, mu, slope):
            return self.sigmoid((x - mu) * slope)
        
        mu_greater = sigmoidmf(X, mean, self.sigmoid_slope)
        temp = self.sigmoid(self.temp)
        mu_great_less = (mu_greater * temp) + (1 - mu_greater) * (1 - temp)


        weight = torch.sigmoid(self.comb_weight)            # (in_features, rules)
        weight = weight.unsqueeze(0)                        # (1, in_features, rules)
        mu = weight * mu_pos_neg + (1 - weight) * mu_great_less

        
        epsilon = 1e-10
        y = torch.log(mu + epsilon)

        max_log_y = torch.max(y, dim=1, keepdim=True)[0]

        y = torch.sum(y - max_log_y, dim=1)

        y = torch.exp(y) * torch.exp(max_log_y.squeeze(dim=1))
        
        return y

    def tsk(self, X, y):
        """
        Coupled TSK: y_i = sum_j [a_ij * (x_j - m_ij)] + b_i
        """
        batch_size = X.shape[0]
        
        means = self.mean  # (in_features, rules)
        
        # Shift inputs by rule centers: (x_j - m_ij)
        X_expanded = X.unsqueeze(1).unsqueeze(3)  
        means_expanded = means.T.unsqueeze(0).unsqueeze(3) 
        
        shifted_inputs = X_expanded - means_expanded
        
        slopes_expanded = self.local_slopes.unsqueeze(0)  

        # linear comb
        linear_terms = torch.matmul(shifted_inputs.transpose(-2, -1), slopes_expanded)
        linear_terms = linear_terms.squeeze(-2) 
        
        # local bias
        rule_outputs = linear_terms + self.local_biases.unsqueeze(0)
        
        # Weight by rule activations and sum
        y = y.reshape(-1, self.rules_count, 1) 
        weighted_outputs = rule_outputs * y 
        
        return weighted_outputs.sum(dim=1) 
    

    def get_interpretable_params(self):
        with torch.no_grad():
            literal = torch.sigmoid(self.literal)
            temp = torch.sigmoid(self.temp)
            weight = torch.sigmoid(self.comb_weight)

            stats = {
                "literal_mean": literal.mean().item(),
                "literal_std": literal.std().item(),
                "temp_mean": temp.mean().item(),
                "temp_std": temp.std().item(),
                "weight_mean": weight.mean().item(),
                "weight_std": weight.std().item(),
                "literal_saturation": ((literal < 0.1) | (literal > 0.9)).float().mean().item(),
                "temp_saturation": ((temp < 0.1) | (temp > 0.9)).float().mean().item(),
                "weight_saturation": ((weight < 0.1) | (weight > 0.9)).float().mean().item(),
                # ADD THESE LINES:
                "slope_mean": self.local_slopes.mean().item(),
                "slope_std": self.local_slopes.std().item(),
                "bias_mean": self.local_biases.mean().item(),
                "center_mean": self.mean.mean().item(),
            }
        return stats


class MamdaniGIFTSHIFT(GIFTSHIFT):
    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool, 
                 drop_out_p=0.5, device=None, dtype=None):
        super().__init__(in_features, rules, out_features, binary, drop_out_p, device, dtype)
    
        factory_kwargs = {'device': device, 'dtype': dtype}
        if binary:
            self.out_features = out_features = 1
        self.mamdani_linear = nn.Linear(rules, out_features, bias=True, **factory_kwargs)

    def mamdani(self, y):
        return self.mamdani_linear(y)
    
    def forward(self, X):
        y = self.encode(X)
        entropy = - y * torch.log(y + 1e-10)

        if self.rules_count > 1:
            y = F.normalize(y, p=1, dim=1)

        reconstructed_X = self.decoder_linear(y)

        y = self.mamdani(y)

        return y, reconstructed_X, entropy


class SklearnGIFTSHIFTWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn wrapper for GIFTSHIFT model
    """
    def __init__(self, model, device=None, dtype=torch.float32):
        self.device = device if device else 'cpu'
        self.dtype = dtype
        self.model = model.to(self.device)

    def fit(self, X, y):
        return self

    def predict(self, X):
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)

        with torch.no_grad():
            y_pred = self.model(X)[0]

        if self.model.binary:
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().numpy() > 0.5
        else:
            y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred.argmax(dim=1).cpu().numpy()
        return y_pred

    def predict_proba(self, X):
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)

        with torch.no_grad():
            y_pred = self.model(X)[0]
        
        if self.model.binary:
            y_pred = torch.sigmoid(y_pred)
            # Convert to (n_samples, 2) format
            neg_proba = 1 - y_pred
            y_pred = torch.cat([neg_proba, y_pred], dim=1)
        else:
            y_pred = torch.softmax(y_pred, dim=1)
    
        return y_pred.cpu().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def _convert_to_tensor(self, data):
        """ Helper function to convert numpy arrays to torch tensors and move to the correct device. """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)

        elif isinstance(data, torch.Tensor):
            data = data.to(self.device)

        elif isinstance(data, pd.DataFrame):
            data = torch.tensor(
                data.values, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(
                "Input data must be a NumPy array or a PyTorch tensor.")
        return data

    def _check_is_filiteraled(self):
        pass

    def get_params(self, deep=True):
        return {
            'model': self.model
        }

    def set_params(self, **parameters):
        return self
    

class GIFTSHIFTENTROPY(GIFTSHIFT):
    """
    GIFT with SHIFT and ENTROPY MINIMIZATION regularization.
    
    Adds a small penalty to reduce entropy per sample, encouraging rule specialization.
    Professor's suggestion: minimize entropy with a VERY SMALL coefficient to prevent collapse.
    """
    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool, 
                 drop_out_p=0.5, entropy_coef=0.001, device=None, dtype=None):
        """
        Args:
            in_features: Number of input features
            rules: Number of fuzzy rules
            out_features: Number of output classes
            binary: Whether this is binary classification
            drop_out_p: Dropout probability
            entropy_coef: Coefficient for entropy penalty (use VERY SMALL, e.g., 0.0001-0.001)
            device: Device to place model on
            dtype: Data type for parameters
        """
        super().__init__(in_features, rules, out_features, binary, drop_out_p, device, dtype)
        self.entropy_coef = entropy_coef
    
    def forward(self, X):
        """
        Forward pass with entropy minimization penalty.
        
        Returns:
            y: Model predictions
            reconstructed_X: Reconstructed input from decoder
            entropy_penalty: Scalar penalty to be added to loss (minimize this!)
        """
        # Encode input to rule activations
        y = self.encode(X)
        
        # Calculate entropy per sample (sum across rules)
        # Lower entropy = sample activates fewer rules (more specialized)
        entropy_per_rule = -y * torch.log(y + 1e-10)      # (batch, rules)
        entropy_per_sample = entropy_per_rule.sum(dim=1)  # (batch,)
        entropy_penalty = entropy_per_sample.mean()       # scalar
        
        # Normalize rule activations if multiple rules
        if self.rules_count > 1:
            y = F.normalize(y, p=1, dim=1)
        
        # Apply dropout
        y = self.drop_out(y)
        
        # Decoder for reconstruction
        reconstructed_X = self.decoder_linear(y)
        
        # TSK inference
        y = self.tsk(X, y)
        
        return y, reconstructed_X, entropy_penalty
    
    def get_entropy_stats(self, X):
        """
        Utility method to analyze entropy distribution.
        Useful for monitoring if collapse is happening.
        
        Returns:
            stats: Dictionary with entropy statistics
        """
        with torch.no_grad():
            y = self.encode(X)
            entropy_per_rule = -y * torch.log(y + 1e-10)
            entropy_per_sample = entropy_per_rule.sum(dim=1)
            
            stats = {
                "mean_entropy": entropy_per_sample.mean().item(),
                "std_entropy": entropy_per_sample.std().item(),
                "min_entropy": entropy_per_sample.min().item(),
                "max_entropy": entropy_per_sample.max().item(),
                "num_active_rules": (y > 0.1).sum(dim=1).float().mean().item(),  # avg rules with >0.1 activation
            }
        return stats


class SklearnGIFTSHIFTENTROPYWrapper(SklearnGIFTSHIFTWrapper):
    """
    Scikit-learn wrapper for GIFTSHIFTENTROPY model.
    Adds entropy analysis capabilities.
    """
    def __init__(self, model, device=None, dtype=torch.float32):
        super().__init__(model, device, dtype)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)

        with torch.no_grad():
            # GIFTSHIFTENTROPY returns 3 values: predictions, reconstruction, entropy_penalty
            y_pred = self.model(X)[0]  # Take first element (predictions)

        if self.model.binary:
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().numpy() > 0.5
        else:
            y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred.argmax(dim=1).cpu().numpy()
        return y_pred

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        """
        self._check_is_filiteraled()
        X = self._convert_to_tensor(X)

        with torch.no_grad():
            y_pred = self.model(X)[0]  # Take first element (predictions)
        
        if self.model.binary:
            y_pred = torch.sigmoid(y_pred)
            # Convert to (n_samples, 2) format
            neg_proba = 1 - y_pred
            y_pred = torch.cat([neg_proba, y_pred], dim=1)
        else:
            y_pred = torch.softmax(y_pred, dim=1)
    
        return y_pred.cpu().numpy()

    def get_entropy_stats(self, X):
        """
        Get entropy statistics for the given input.
        Useful for monitoring rule specialization and preventing collapse.
        
        Args:
            X: Input data (numpy array, pandas DataFrame, or torch tensor)
            
        Returns:
            Dictionary with entropy statistics
        """
        X = self._convert_to_tensor(X)
        
        with torch.no_grad():
            # Get rule activations without going through full forward pass
            y = self.model.encode(X)
            entropy_per_rule = -y * torch.log(y + 1e-10)
            entropy_per_sample = entropy_per_rule.sum(dim=1)
            
            stats = {
                "mean_entropy_per_sample": entropy_per_sample.mean().item(),
                "std_entropy_per_sample": entropy_per_sample.std().item(),
                "min_entropy_per_sample": entropy_per_sample.min().item(),
                "max_entropy_per_sample": entropy_per_sample.max().item(),
                "mean_active_rules": (y > 0.1).sum(dim=1).float().mean().item(),
                "total_rules": self.model.rules_count,
                "entropy_coef": self.model.entropy_coef,
            }
        return stats

    def get_rule_activations(self, X):
        """
        Get raw rule activations for interpretability.
        
        Args:
            X: Input data
            
        Returns:
            rule_activations: numpy array of shape (n_samples, n_rules)
        """
        X = self._convert_to_tensor(X)
        
        with torch.no_grad():
            y = self.model.encode(X)
            if self.model.rules_count > 1:
                y = F.normalize(y, p=1, dim=1)
        
        return y.cpu().numpy()

    def score_with_entropy_analysis(self, X, y):
        """
        Calculate accuracy and also return entropy statistics.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            accuracy: Accuracy score
            entropy_stats: Dictionary with entropy statistics
        """
        accuracy = self.score(X, y)
        entropy_stats = self.get_entropy_stats(X)
        entropy_stats["accuracy"] = accuracy
        return entropy_stats


# Also create wrapper for Mamdani version if needed
class SklearnMamdaniGIFTSHIFTENTROPYWrapper(SklearnGIFTSHIFTENTROPYWrapper):
    """
    Scikit-learn wrapper for MamdaniGIFTSHIFTENTROPY model.
    """
    def __init__(self, model, device=None, dtype=torch.float32):
        super().__init__(model, device, dtype)

class MamdaniGIFTSHIFTENTROPY(GIFTSHIFTENTROPY):
    """
    Mamdani version of GIFTSHIFT with entropy minimization.
    Uses Mamdani inference instead of TSK.
    """
    def __init__(self, in_features: int, rules: int, out_features: int, binary: bool, 
                 drop_out_p=0.5, entropy_coef=0.001, device=None, dtype=None):
        super().__init__(in_features, rules, out_features, binary, drop_out_p, entropy_coef, device, dtype)
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        if binary:
            self.out_features = out_features = 1
        self.mamdani_linear = nn.Linear(rules, out_features, bias=True, **factory_kwargs)
    
    def mamdani(self, y):
        return self.mamdani_linear(y)
    
    def forward(self, X):
        # Encode input to rule activations
        y = self.encode(X)
        
        # Calculate entropy penalty (minimize this!)
        entropy_per_rule = -y * torch.log(y + 1e-10)
        entropy_per_sample = entropy_per_rule.sum(dim=1)
        entropy_penalty = entropy_per_sample.mean()
        
        # Normalize and dropout
        if self.rules_count > 1:
            y = F.normalize(y, p=1, dim=1)
        
        y = self.drop_out(y)
        reconstructed_X = self.decoder_linear(y)
        
        # Mamdani inference
        y = self.mamdani(y)
        
        return y, reconstructed_X, entropy_penalty