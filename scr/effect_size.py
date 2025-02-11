from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import pearsonr
import numpy as np


def d_value_calculation(group1, group2):
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)
    pooled_std = np.sqrt(((std1 ** 2) + (std2 ** 2)) / 2)
    return (mean1 - mean2) / pooled_std

def interpret_cohens_d(d_value):
    if abs(d_value) < 0.2:
        interpretation = "Small effect size"
        explanation = "There is a small difference between the groups, and the effect is minimal."
    elif 0.2 <= abs(d_value) < 0.5:
        interpretation = "Medium effect size"
        explanation = "The difference between the groups is moderate, with noticeable effects."
    elif 0.5 <= abs(d_value) < 0.8:
        interpretation = "Large effect size"
        explanation = "There is a large difference between the groups, with a strong effect."
    else:
        interpretation = "Very large effect size"
        explanation = "The difference between the groups is very large, indicating a very strong effect."
    return interpretation, explanation


def calculate_effect_sizes(df, group_column, Metrics):
    effect_size_results = []
    for column in Metrics:
        group_values = df[group_column].unique()  # Get unique groups in the 'group_column'
        for i in range(len(group_values)):
            for j in range(i + 1, len(group_values)):  # Ensure each pair is unique
                group1 = df[df[group_column] == group_values[i]][column]
                group2 = df[df[group_column] == group_values[j]][column]
                
                # Calculate Cohen's d for this pair
                d_value = d_value_calculation(group1, group2)
                interpretation, explanation = interpret_cohens_d(d_value)
                
                # Store results in a list
                effect_size_results.append({
                    'Column': column,
                    'Pair': f'{group_values[i]} vs {group_values[j]}',
                    'Cohen\'s d': d_value,
                    'Interpretation': interpretation,
                    'Explanation': explanation
                })

    # Create a DataFrame from the results
    effect_size_df = pd.DataFrame(effect_size_results)
    return effect_size_df

def cohens_d(df, Metrics, group_column="species"):
    effect_sizes_df = calculate_effect_sizes(df, group_column="species", Metrics=Metrics)
    pd.set_option('display.max_colwidth', 120) 
    return effect_sizes_df






def compute_pearson_r(df, Metrics):
    results = []

    for i, col1 in enumerate(Metrics):
        for col2 in Metrics[i+1:]:
            r_value, p_value = pearsonr(df[col1], df[col2])

            direction = ("Positive" if r_value > 0 else 
                         "Negative" if r_value < 0 else "No correlation")
            strength = ("Strong" if abs(r_value) >= 0.7 else 
                        "Moderate" if abs(r_value) >= 0.3 else "Weak")

            results.append({
                'Variable 1': col1, 'Variable 2': col2,
                'Pearson\'s r': r_value, 'P-value': p_value,
                'Direction': direction, 'Strength': strength
            })
    
    return pd.DataFrame(results)



def compute_partial_eta_squared(df, numerical_columns, groups):
    results = []
    for column in numerical_columns:
        
        for factor in groups:
            safe_column_name = column.replace("(", "").replace(")", "").replace("/", "").replace(" ", "_")
            df = df.rename(columns={column: safe_column_name})
            formula = f'{safe_column_name} ~ C({factor})'
            model = ols(formula, data=df).fit()
            anova_results = anova_lm(model, typ=2)
            SS_factor = anova_results['sum_sq'][f'C({factor})']
            SS_error = anova_results['sum_sq']['Residual']
            partial_eta_squared = SS_factor / (SS_factor + SS_error)
            
            results.append({
                "Variables": column,
                "Factor": factor,
                "Partial Eta-squared (ηp²)": partial_eta_squared
            })
    

    eta_squared_df = pd.DataFrame(results)
    
    def interpret_eta_squared(eta_squared):
        if eta_squared >= 0.14: return "Large effect size (≥ 14%)"
        elif eta_squared >= 0.06: return "Medium effect size (6% - 14%)"
        else: return "Small effect size (< 6%)"
    eta_squared_df['Interpretation'] = eta_squared_df['Partial Eta-squared (ηp²)'].apply(interpret_eta_squared)
    return eta_squared_df



def compute_eta_squared(df, Metrics, groups):

    def perform_anova(df, dependent_var, groups):
        """Perform ANOVA and return results in a DataFrame."""
        model = ols(f'{dependent_var} ~ C({groups})', data=df).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        
        eta_sq = eta_squared(aov_table)
        
        # Create a DataFrame including eta-squared (η²) as a new column
        aov_table["Eta-squared (η²)"] = np.nan
        aov_table.loc[f'C({groups})', "Eta-squared (η²)"] = eta_sq  # Assign only to between-group row
        
        return aov_table.reset_index().rename(columns={"index": "Source"})  # Reset index for cleaner output
    
    def eta_squared(aov_table):
        """Calculate eta-squared (η²) from the ANOVA table."""
        ss_between = aov_table["sum_sq"].iloc[0]  # Use .iloc[0] to avoid FutureWarning
        ss_total = aov_table["sum_sq"].sum()  # Total sum of squares
        return ss_between / ss_total
     
    
    # Example loop to process multiple metrics
    results = []  # Store results for all metrics
    
    for Metric in Metrics:
        safe_column_name = Metric.replace("(", "").replace(")", "").replace("/", "").replace(" ", "_")
        data = df.rename(columns={Metric: safe_column_name})
          
        anova_df = perform_anova(data, safe_column_name, groups)
        anova_df.insert(0, "Metric", Metric)  # Add Metric column for tracking
        
        results.append(anova_df)
    
    # Combine all results into a single DataFrame
    Eta_squared_df = pd.concat(results, ignore_index=True)
    return Eta_squared_df

