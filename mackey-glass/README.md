## Mackey-Glass Regression with ANFIS

This project explores ANFIS-based regression on the Mackey-Glass time series with incremental improvements:

1. **Simple ANFIS**  
   - Implemented a basic ANFIS model to perform regression on the Mackey-Glass dataset.  
   - Serves as the baseline neuro-fuzzy model.

2. **FCM Initialization**  
   - Improved the initial membership functions using Fuzzy C-Means (FCM) clustering.  
   - This helps ANFIS converge faster and achieve better accuracy.

3. **Advanced Membership Functions**  
   - Incorporated the core idea from the reference paper: blending Gaussian and Sigmoid MFs.  
   - The blended MF is computed as:  
     $$
     \mu^+ = s \, \mu_{\text{gauss}} + (1-s) \, \mu_{\text{sig}}
     $$  
   - This allows the model to adaptively combine smooth and threshold-like behaviors for better function approximation.
