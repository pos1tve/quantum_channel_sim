import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from typing import Dict, List, Tuple
import seaborn as sns


class AdditivityStatisticalModels:
    """Advanced statistical models for analyzing Holevo capacity additivity."""
    
    def __init__(self, results: List[Dict]):
        """Initialize with test results."""
        self.results = results
        self.df = pd.DataFrame(results)
        self._prepare_features()
    
    def _prepare_features(self):
        """Prepare features for statistical modeling."""
        # Extract channel parameters if available
        features = []
        
        for result in self.results:
            feature_dict = {}
            
            # Channel 1 parameters
            if 'channel1_p' in result:
                feature_dict['ch1_p'] = result['channel1_p']
            if 'channel1_gamma' in result:
                feature_dict['ch1_gamma'] = result['channel1_gamma']
                
            # Channel 2 parameters
            if 'channel2_p' in result:
                feature_dict['ch2_p'] = result['channel2_p']
            if 'channel2_gamma' in result:
                feature_dict['ch2_gamma'] = result['channel2_gamma']
                
            # Individual capacities
            feature_dict['chi1'] = result['chi1']
            feature_dict['chi2'] = result['chi2']
            feature_dict['chi_sum'] = result['chi1'] + result['chi2']
            feature_dict['chi_product'] = result['chi1'] * result['chi2']
            feature_dict['chi_min'] = min(result['chi1'], result['chi2'])
            feature_dict['chi_max'] = max(result['chi1'], result['chi2'])
            
            # Channel type encoding
            feature_dict['ch1_is_depol'] = int(result['channel1_type'] == 'DepolarizingChannel')
            feature_dict['ch1_is_ad'] = int(result['channel1_type'] == 'AmplitudeDampingChannel')
            feature_dict['ch1_is_erasure'] = int(result['channel1_type'] == 'ErasureChannel')
            feature_dict['ch2_is_depol'] = int(result['channel2_type'] == 'DepolarizingChannel')
            feature_dict['ch2_is_ad'] = int(result['channel2_type'] == 'AmplitudeDampingChannel')
            feature_dict['ch2_is_erasure'] = int(result['channel2_type'] == 'ErasureChannel')
            feature_dict['same_channel_type'] = int(result['channel1_type'] == result['channel2_type'])
            
            features.append(feature_dict)
        
        self.features_df = pd.DataFrame(features)
        self.features_df = self.features_df.fillna(0)  # Fill NaN with 0
        
    def regression_analysis(self) -> Dict:
        """Perform regression analysis to predict violations."""
        print("\nPerforming regression analysis...")
        
        # Prepare data
        X = self.features_df.values
        y = self.df['violation'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Random Forest Regression
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = rf_model.predict(X_train)
        y_pred_test = rf_model.predict(X_test)
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.features_df.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'][:10], 
                feature_importance['importance'][:10])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features for Predicting Violations')
        plt.tight_layout()
        plt.show()
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': feature_importance,
            'model': rf_model
        }
    
    def bayesian_analysis(self) -> Dict:
        """Perform Bayesian analysis of additivity violations."""
        print("\nPerforming Bayesian analysis...")
        
        violations = self.df['violation'].values
        
        # Fit normal distribution (conjugate prior)
        mu_prior = 0  # Prior mean (perfect additivity)
        sigma_prior = 0.1  # Prior standard deviation
        
        # Update with data (using conjugate prior)
        n = len(violations)
        sample_mean = np.mean(violations)
        sample_var = np.var(violations, ddof=1)
        
        # Posterior parameters
        sigma_posterior = 1 / (1/sigma_prior**2 + n/sample_var)
        mu_posterior = sigma_posterior * (mu_prior/sigma_prior**2 + n*sample_mean/sample_var)
        
        # Credible interval
        credible_interval = stats.norm.interval(0.95, mu_posterior, np.sqrt(sigma_posterior))
        
        # Bayes factor for H0: μ = 0 vs H1: μ ≠ 0
        # Using Savage-Dickey ratio
        prior_density_at_zero = stats.norm.pdf(0, mu_prior, sigma_prior)
        posterior_density_at_zero = stats.norm.pdf(0, mu_posterior, np.sqrt(sigma_posterior))
        bayes_factor = posterior_density_at_zero / prior_density_at_zero
        
        # Plot posterior distribution
        x = np.linspace(mu_posterior - 4*np.sqrt(sigma_posterior), 
                       mu_posterior + 4*np.sqrt(sigma_posterior), 1000)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, stats.norm.pdf(x, mu_prior, sigma_prior), 
                'b--', label='Prior', alpha=0.5)
        plt.plot(x, stats.norm.pdf(x, mu_posterior, np.sqrt(sigma_posterior)), 
                'r-', label='Posterior', linewidth=2)
        plt.axvline(x=0, color='black', linestyle=':', label='Perfect additivity')
        plt.fill_between(x, 0, stats.norm.pdf(x, mu_posterior, np.sqrt(sigma_posterior)),
                        where=(x >= credible_interval[0]) & (x <= credible_interval[1]),
                        alpha=0.3, color='red', label='95% Credible Interval')
        plt.xlabel('Mean Violation')
        plt.ylabel('Density')
        plt.title('Bayesian Analysis of Additivity Violations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return {
            'posterior_mean': mu_posterior,
            'posterior_std': np.sqrt(sigma_posterior),
            'credible_interval': credible_interval,
            'bayes_factor': bayes_factor,
            'evidence_interpretation': self._interpret_bayes_factor(bayes_factor)
        }
    
    def _interpret_bayes_factor(self, bf: float) -> str:
        """Interpret Bayes factor according to Jeffreys' scale."""
        if bf > 100:
            return "Decisive evidence for H0 (perfect additivity)"
        elif bf > 10:
            return "Strong evidence for H0"
        elif bf > 3:
            return "Moderate evidence for H0"
        elif bf > 1:
            return "Weak evidence for H0"
        elif bf > 1/3:
            return "Weak evidence for H1 (violations)"
        elif bf > 1/10:
            return "Moderate evidence for H1"
        elif bf > 1/100:
            return "Strong evidence for H1"
        else:
            return "Decisive evidence for H1 (systematic violations)"
    
    def mixture_model_analysis(self) -> Dict:
        """Fit Gaussian mixture model to detect different violation regimes."""
        from sklearn.mixture import GaussianMixture
        
        print("\nPerforming mixture model analysis...")
        
        violations = self.df['violation'].values.reshape(-1, 1)
        
        # Try different numbers of components
        n_components_range = range(1, 6)
        bic_scores = []
        aic_scores = []
        models = []
        
        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(violations)
            bic_scores.append(gmm.bic(violations))
            aic_scores.append(gmm.aic(violations))
            models.append(gmm)
        
        # Select best model using BIC
        best_n = n_components_range[np.argmin(bic_scores)]
        best_model = models[np.argmin(bic_scores)]
        
        # Plot model selection
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(n_components_range, bic_scores, 'bo-', label='BIC')
        ax1.plot(n_components_range, aic_scores, 'ro-', label='AIC')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Information Criterion')
        ax1.set_title('Model Selection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot best fit
        x_range = np.linspace(violations.min(), violations.max(), 1000).reshape(-1, 1)
        
        # Plot histogram
        ax2.hist(violations, bins=50, density=True, alpha=0.5, color='gray')
        
        # Plot individual components
        for i in range(best_n):
            mean = best_model.means_[i, 0]
            std = np.sqrt(best_model.covariances_[i, 0, 0])
            weight = best_model.weights_[i]
            
            y = weight * stats.norm.pdf(x_range.ravel(), mean, std)
            ax2.plot(x_range, y, label=f'Component {i+1}')
        
        # Plot total
        y_total = np.exp(best_model.score_samples(x_range)) * len(violations) / 50
        ax2.plot(x_range, y_total, 'k-', linewidth=2, label='Total')
        
        ax2.set_xlabel('Violation')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Gaussian Mixture Model (n={best_n})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Classify violations
        labels = best_model.predict(violations)
        
        return {
            'optimal_components': best_n,
            'means': best_model.means_.ravel(),
            'stds': np.sqrt(best_model.covariances_.ravel()),
            'weights': best_model.weights_,
            'labels': labels,
            'bic': bic_scores[best_n-1],
            'aic': aic_scores[best_n-1]
        }
    
    def extreme_value_analysis(self) -> Dict:
        """Analyze extreme violations using extreme value theory."""
        from scipy.stats import genextreme
        
        print("\nPerforming extreme value analysis...")
        
        violations = self.df['violation'].values
        
        # Fit GEV distribution to maximum violations
        # Block maxima approach
        block_size = 10
        n_blocks = len(violations) // block_size
        block_maxima = []
        
        for i in range(n_blocks):
            block = violations[i*block_size:(i+1)*block_size]
            block_maxima.append(np.max(np.abs(block)))
        
        # Fit GEV
        params = genextreme.fit(block_maxima)
        shape, loc, scale = params
        
        # Calculate return levels
        return_periods = [10, 50, 100, 500]
        return_levels = {}
        
        for T in return_periods:
            p = 1 - 1/T
            return_level = genextreme.ppf(p, shape, loc, scale)
            return_levels[f'{T}-test'] = return_level
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # QQ plot
        theoretical_quantiles = genextreme.ppf(
            np.linspace(0.01, 0.99, len(block_maxima)), shape, loc, scale
        )
        empirical_quantiles = np.sort(block_maxima)
        
        ax1.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.6)
        ax1.plot([0, max(theoretical_quantiles)], [0, max(theoretical_quantiles)], 
                'r--', label='y=x')
        ax1.set_xlabel('Theoretical Quantiles (GEV)')
        ax1.set_ylabel('Empirical Quantiles')
        ax1.set_title('QQ Plot for Extreme Value Distribution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Return level plot
        T_range = np.logspace(0, 3, 100)
        return_level_curve = [genextreme.ppf(1-1/T, shape, loc, scale) for T in T_range]
        
        ax2.semilogx(T_range, return_level_curve, 'b-', linewidth=2)
        for T, level in return_levels.items():
            ax2.scatter([int(T.split('-')[0])], [level], s=100, c='red', zorder=5)
            ax2.annotate(T, (int(T.split('-')[0]), level), xytext=(5, 5), 
                        textcoords='offset points')
        
        ax2.set_xlabel('Return Period (number of tests)')
        ax2.set_ylabel('Return Level (max |violation|)')
        ax2.set_title('Return Level Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'gev_shape': shape,
            'gev_location': loc,
            'gev_scale': scale,
            'return_levels': return_levels,
            'interpretation': self._interpret_gev_shape(shape)
        }
    
    def _interpret_gev_shape(self, shape: float) -> str:
        """Interpret GEV shape parameter."""
        if shape < -0.5:
            return "Heavy lower tail (bounded maximum violations)"
        elif shape < 0:
            return "Light lower tail (Weibull type)"
        elif shape == 0:
            return "Exponential tail (Gumbel type)"
        elif shape < 0.5:
            return "Light upper tail (Fréchet type)"
        else:
            return "Heavy upper tail (potential for extreme violations)"
    
    def correlation_analysis(self) -> Dict:
        """Analyze correlations between channel properties and violations."""
        print("\nPerforming correlation analysis...")
        
        # Combine features with violations
        analysis_df = self.features_df.copy()
        analysis_df['violation'] = self.df['violation']
        analysis_df['relative_violation'] = self.df['relative_violation']
        
        # Calculate correlations
        correlations = analysis_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlations), k=1)
        sns.heatmap(correlations, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, 
                   cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Channel Properties and Violations')
        plt.tight_layout()
        plt.show()
        
        # Find strongest correlations with violations
        violation_corr = correlations['violation'].drop('violation').abs().sort_values(ascending=False)
        
        return {
            'correlation_matrix': correlations,
            'top_correlations': violation_corr.head(10),
            'significant_predictors': violation_corr[violation_corr > 0.3].index.tolist()
        }
    
    def generate_statistical_report(self) -> str:
        """Generate comprehensive statistical analysis report."""
        report = """
STATISTICAL ANALYSIS OF HOLEVO CAPACITY ADDITIVITY
==================================================

"""
        
        # Basic statistics
        report += f"""
1. BASIC STATISTICS
-------------------
Total samples: {len(self.df)}
Mean violation: {self.df['violation'].mean():.6f}
Median violation: {self.df['violation'].median():.6f}
Standard deviation: {self.df['violation'].std():.6f}
Skewness: {self.df['violation'].skew():.3f}
Kurtosis: {self.df['violation'].kurtosis():.3f}

"""
        
        # Bayesian analysis
        bayes_results = self.bayesian_analysis()
        report += f"""
2. BAYESIAN ANALYSIS
--------------------
Posterior mean: {bayes_results['posterior_mean']:.6f}
Posterior std: {bayes_results['posterior_std']:.6f}
95% Credible interval: [{bayes_results['credible_interval'][0]:.6f}, {bayes_results['credible_interval'][1]:.6f}]
Bayes factor: {bayes_results['bayes_factor']:.3f}
Interpretation: {bayes_results['evidence_interpretation']}

"""
        
        # Mixture model
        mixture_results = self.mixture_model_analysis()
        report += f"""
3. MIXTURE MODEL ANALYSIS
-------------------------
Optimal number of components: {mixture_results['optimal_components']}
Component means: {mixture_results['means']}
Component weights: {mixture_results['weights']}
BIC: {mixture_results['bic']:.2f}

"""
        
        # Extreme value analysis
        extreme_results = self.extreme_value_analysis()
        report += f"""
4. EXTREME VALUE ANALYSIS
-------------------------
GEV shape parameter: {extreme_results['gev_shape']:.3f}
Interpretation: {extreme_results['interpretation']}
Expected maximum violation in 100 tests: {extreme_results['return_levels']['100-test']:.6f}

"""
        
        # Regression analysis
        regression_results = self.regression_analysis()
        report += f"""
5. PREDICTIVE MODELING
----------------------
Random Forest R² (test set): {regression_results['test_r2']:.3f}
Top 3 predictive features:
"""
        for i, row in regression_results['feature_importance'].head(3).iterrows():
            report += f"  - {row['feature']}: {row['importance']:.3f}\n"
        
        # Correlation analysis
        correlation_results = self.correlation_analysis()
        report += f"""
6. CORRELATION ANALYSIS
-----------------------
Significant predictors of violations:
"""
        for predictor in correlation_results['significant_predictors'][:5]:
            corr = correlation_results['correlation_matrix'].loc[predictor, 'violation']
            report += f"  - {predictor}: r = {corr:.3f}\n"
        
        return report


def perform_advanced_statistical_analysis(results: List[Dict]) -> Dict:
    """Perform comprehensive statistical analysis on test results."""
    analyzer = AdditivityStatisticalModels(results)
    
    # Run all analyses
    all_results = {
        'bayesian': analyzer.bayesian_analysis(),
        'mixture_model': analyzer.mixture_model_analysis(),
        'extreme_value': analyzer.extreme_value_analysis(),
        'regression': analyzer.regression_analysis(),
        'correlation': analyzer.correlation_analysis()
    }
    
    # Generate report
    report = analyzer.generate_statistical_report()
    all_results['report'] = report
    
    return all_results 