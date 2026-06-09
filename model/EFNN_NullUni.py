import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import pandas as pd

class EFNN_NullUni(nn.Module):
    """
    EFNN-NullUni: Evolving Fuzzy Neural Network based on Null-Uninorm.
    این مدل شروط را با گیت‌های انعطاف‌پذیر Null-Uninorm ترکیب می‌کند (ترکیب AND/OR).
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

        # پارامترهای توابع عضویت گوسی (مراکز و انحراف معیارها)
        self.mean = nn.Parameter(torch.rand((in_features, rules), **factory_kwargs))
        self.std = nn.Parameter(torch.rand((in_features, rules), **factory_kwargs))
        
        # وزن‌های اهمیت ویژگی‌ها (برای اعمال رویکرد شبیه به وزن‌دهی ویژگی‌ها در مقاله)
        self.feature_weights = nn.Parameter(torch.ones((in_features, rules), **factory_kwargs))

        # اصل نوآوری: پارامتر اتصال گیت Null-Uninorm برای هر قانون
        # این پارامتر تعیین می‌کند شرط قانون AND باشد یا OR یا ترکیبی
        self.g = nn.Parameter(torch.randn((rules,), **factory_kwargs) * 0.1)

        # لایه خروجی مامدانی (Mamdani Consequent) برای استخراج مستقیم کلاس‌ها
        self.mamdani_linear = nn.Linear(in_features=rules, out_features=out_features, bias=True, **factory_kwargs)
        
        # لایه دکودر برای حفظ ساختار GIFT شما (کمک به پایداری ویژگی‌ها)
        self.decoder_linear = nn.Linear(in_features=rules, out_features=in_features, bias=True, **factory_kwargs)

        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(p=drop_out_p)

    def forward(self, X):
        y = self.encode(X)
        entropy = - y * torch.log(y + 1e-10)

        if self.rules_count > 1:
            y = F.normalize(y, p=1, dim=1)

        y = self.drop_out(y)

        # بازسازی ورودی (مشابه ساختار اتوانکودر GIFT شما)
        reconstructed_X = self.decoder_linear(y)

        # خروجی مامدانی (مشابه الگوی مقاله)
        y_out = self.mamdani_linear(y)

        return y_out, reconstructed_X, entropy

    def encode(self, X):
        mean = self.mean.view(1, *self.mean.shape)
        std = F.softplus(self.std).view(1, *self.std.shape)
        
        X = X.view(*X.shape, 1)

        # تابع عضویت گوسی
        def gaussmf(x, mu, sigma):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        mu_matrix = gaussmf(X, mean, std) # Shape: (batch, in_features, rules)

        # اعمال وزن ویژگی‌ها (کاهش اثر ویژگی‌های نویز)
        w = self.sigmoid(self.feature_weights).unsqueeze(0)
        mu_matrix = mu_matrix * w

        # پیاده‌سازی ساده‌شده ریاضی گیت Null-Uninorm روی ویژگی‌ها (بعد dim=1)
        # پارامتر g نوع پیوند را مشخص می‌کند (g نزدیک به 0 یعنی AND، نزدیک به 1 یعنی OR)
        g_param = self.sigmoid(self.g).view(1, 1, self.rules_count)
        
        # بخش AND (حاصلضرب درجات عضویت)
        and_part = torch.prod(mu_matrix + 1e-8, dim=1)
        # بخش OR (ماکزیمم یا جمع فازی درجات عضویت)
        or_part = torch.max(mu_matrix, dim=1)[0]

        # ترکیب وزنی توافقی بین AND و OR براساس پارامتر آموخته‌شده g
        y = (1 - g_param.squeeze(1)) * and_part + g_param.squeeze(1) * or_part
        
        return y

    def get_interpretable_params(self):
        """
        متدی کاملاً همگام با بخش get_interpretable_params شما برای ارزیابی تفسیرپذیری
        """
        with torch.no_grad():
            g_val = self.sigmoid(self.g)
            f_weights = self.sigmoid(self.feature_weights)
            
            stats = {
                "g_connector_mean": g_val.mean().item(),
                "g_connector_std": g_val.std().item(),
                # نشان می‌دهد چند درصد قوانین تمایل به OR خالص (>0.8) یا AND خالص (<0.2) پیدا کرده‌اند
                "g_or_saturation": (g_val > 0.8).float().mean().item(),
                "g_and_saturation": (g_val < 0.2).float().mean().item(),
                "feature_weights_mean": f_weights.mean().item(),
                # ویژگی‌های حذف شده یا نادیده گرفته شده (وزن‌های نزدیک به صفر)
                "pruned_features_ratio": (f_weights < 0.1).float().mean().item(),
                "center_mean": self.mean.mean().item(),
            }
        return stats


class SklearnEFNNWrapper(BaseEstimator, ClassifierMixin):
    """
    رابط Scikit-Learn برای مدل EFNN_NullUni (عیناً هماهنگ با ساختار کدهای خودت)
    """
    def __init__(self, model, device=None, dtype=torch.float32):
        self.device = device if device else 'cpu'
        self.dtype = dtype
        self.model = model.to(self.device)

    def fit(self, X, y):
        return self

    def predict(self, X):
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
        X = self._convert_to_tensor(X)
        with torch.no_grad():
            y_pred = self.model(X)[0]
        
        if self.model.binary:
            y_pred = torch.sigmoid(y_pred)
            neg_proba = 1 - y_pred
            y_pred = torch.cat([neg_proba, y_pred], dim=1)
        else:
            y_pred = torch.softmax(y_pred, dim=1)
    
        return y_pred.cpu().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def _convert_to_tensor(self, data):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
        elif isinstance(data, torch.Tensor):
            data = data.to(self.device)
        elif isinstance(data, pd.DataFrame):
            data = torch.tensor(data.values, dtype=torch.float32, device=self.device)
        else:
            raise ValueError("Input data must be a NumPy array, Pandas DataFrame or a PyTorch tensor.")
        return data