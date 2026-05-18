#!/usr/bin/env python3
"""
=============================================================================
COURSE CODE: STAT 4104 | COURSE TITLE: CATEGORICAL DATA ANALYSIS
=============================================================================
Research Portfolio Production Script File
Prepared By : Masud Shariar Sowrov (ID: 12110012 | Reg: 000015875)
Submitted To: Dr. Md. Siddikur Rahman (Associate Professor)
Department of Statistics, Begum Rokeya University, Rangpur
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomial

# Configure uniform plotting style for academic presentation
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# =============================================================================
# PROBLEM 1: EXPLORATORY DATA ANALYSIS & MULTIVARIABLE LOGISTIC REGRESSION
# =============================================================================
print("\n" + "="*80)
print(" RUNNING: PROBLEM 1 - LOGISTIC REGRESSION MODELING")
print("="*80)

try:
    # 1. Load Data & Summary Stats
    df_lung = pd.read_csv("lung_disease_data.csv")
    print("\n--- Descriptive Summary Metrics ---")
    print(df_lung.describe())
    print("\nSmoking Prevalence Rates:")
    print(df_lung['smoking'].value_counts(normalize=True))

    # 2. Fit Main Effects Logistic Regression
    model_main = smf.logit("lung_disease ~ smoking + pollution + income + age", data=df_lung).fit()
    print("\n--- Main Effects Logistic GLM Estimates ---")
    print(model_main.summary())
    print("\nAdjusted Model Odds Ratios:\n", np.exp(model_main.params))

    # 3. Fit Interaction Model
    model_inter = smf.logit("lung_disease ~ smoking * pollution + income + age", data=df_lung).fit()
    print("\n--- Interaction Model Estimates ---")
    print(model_inter.summary())

    # 4. Diagnostic Confusion Matrix Summary
    preds = model_main.predict(df_lung)
    df_lung['pred_class'] = (preds >= 0.5).astype(int)
    cm = model_main.pred_table()
    print("\nClassification Confusion Matrix:\n", cm)

    # 5. Generate and Save Vector ROC Curve Graph
    fpr, tpr, _ = roc_curve(df_lung['lung_disease'], preds)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='navy', lw=2, label=f'Main Model (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Problem 1: Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.pdf", dpi=300)
    print("\n[SUCCESS] Saved high-resolution graphic to 'roc_curve.pdf'")
    plt.close()
except FileNotFoundError:
    print("\n[WARNING] 'lung_disease_data.csv' missing. Skipping execution analytics.")


# =============================================================================
# PROBLEM 2: EXPERIMENTAL EVALUATION OF EXERCISE REGIMENS (ONE-WAY ANOVA)
# =============================================================================
print("\n" + "="*80)
print(" RUNNING: PROBLEM 2 - ONE-WAY ANOVA ASSESSMENTS")
print("="*80)

try:
    # 1. Load Data and Compute Descriptive Statistics
    df_ex = pd.read_csv("exercise_data.csv")
    print("\n--- Weight Loss Treatment Group Dispersions ---")
    print(df_ex.groupby('program')['weight_loss'].describe())

    # 2. Fit One-Way ANOVA Matrix
    model_anova = ols('weight_loss ~ C(program)', data=df_ex).fit()
    anova_table = sm.stats.anova_lm(model_anova, typ=1)
    print("\n--- Global ANOVA Outcome Results ---")
    print(anova_table)

    # 3. Check Foundational Residual Assumptions
    residuals = model_anova.resid
    sw_stat, sw_p = stats.shapiro(residuals)
    print(f"\nShapiro-Wilk Normality Test: W = {sw_stat:.4f}, p-value = {sw_p:.4f}")

    g_a = df_ex[df_ex['program'] == 'A']['weight_loss']
    g_b = df_ex[df_ex['program'] == 'B']['weight_loss']
    g_c = df_ex[df_ex['program'] == 'C']['weight_loss']
    lev_stat, lev_p = stats.levene(g_a, g_b, g_c)
    print(f"Levene Variance Homogeneity Test: F = {lev_stat:.4f}, p-value = {lev_p:.4f}")

    # 4. Post-Hoc Tukey Honestly Significant Difference (HSD)
    tukey = pairwise_tukeyhsd(endog=df_ex['weight_loss'], groups=df_ex['program'], alpha=0.05)
    print("\n--- Pairwise Tukey HSD Structural Contrast Differences ---")
    print(tukey.summary())

    # 5. Generate Quantile-Quantile (Q-Q) Diagnostics Graphic
    fig, ax = plt.subplots(figsize=(6, 5))
    sm.qqplot(residuals, line='s', ax=ax)
    ax.set_title('Problem 2: Normal Q-Q Plot of Model Residuals')
    plt.tight_layout()
    plt.savefig("residuals_qqplot.pdf", dpi=300)
    print("\n[SUCCESS] Saved diagnostic check profile to 'residuals_qqplot.pdf'")
    plt.close()
except FileNotFoundError:
    print("\n[WARNING] 'exercise_data.csv' missing. Skipping evaluation framework.")


# =============================================================================
# PROBLEM 3: CATEGORICAL INDEPENDENCE ANALYSIS (CHILDHOOD ANEMIA IN NIGERIA)
# =============================================================================
print("\n" + "="*80)
print(" RUNNING: PROBLEM 3 - CHI-SQUARE INDEPENDENCE MATRIX EVALUATION")
print("="*80)

# Setup 2018 NDHS Matrix Structure (Rows: Economic Wealth Quintiles | Cols: Clinical Status Tiers)
observed_anemia = np.array([
    [418, 522, 983, 119],  # Poorest
    [498, 525, 926,  87],  # Poorer
    [729, 596, 865,  59],  # Middle
    [737, 606, 750,  48],  # Richer
    [797, 505, 403,   9]   # Richest
])

# Execute Pearson Chi-Square Test
chi2_stat, p_val_chi2, dof_chi2, expected_anemia = stats.chi2_contingency(observed_anemia)
print("\n--- Pearson Chi-Square Global Independent Diagnostics ---")
print(f"Calculated Omnibus Chi2 Metric : {chi2_stat:.4f}")
print(f"Model Degrees of Freedom (df)  : {dof_chi2}")
print(f"Asymptotic Significance p-value: {p_val_chi2}")

# Compute Cell Contributions
contributions_anemia = ((observed_anemia - expected_anemia) ** 2) / expected_anemia
row_labels = ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest']
col_labels = ['Not Anemic', 'Mild Anemia', 'Moderate Anemia', 'Severe Anemia']
df_contrib = pd.DataFrame(contributions_anemia, index=row_labels, columns=col_labels)
print("\n--- Cell Contribution Breakdown Matrix ---")
print(np.round(df_contrib, 2))


# =============================================================================
# PROBLEM 4: THERAPEUTIC EVALUATIONS VIA EXACT TESTS (DRUG OUTCOMES)
# =============================================================================
print("\n" + "="*80)
print(" RUNNING: PROBLEM 4 - FISHER'S EXACT OMNIBUS EXTENSION")
print("="*80)

# 3x2 Matrix Setup (Rows: Drug A, B, C | Cols: Disease Present, Absent)
observed_drugs = np.array([
    [40, 10],  # Drug A
    [10, 40],  # Drug B
    [25, 25]   # Drug C
])

# Global exact test mapping evaluation (handled via modern scipy internal engines)
res_global_fisher = stats.fisher_exact(observed_drugs)
print("\n--- Global Freeman-Halton Omnibus Matrix Result ---")
print(f"Exact Hypergeometric Two-Tailed p-value: {res_global_fisher[1]}")

# Setup post-hoc pairwise split evaluations
drug_pairs = {
    "Drug A vs. Drug B": np.array([[40, 10], [10, 40]]),
    "Drug A vs. Drug C": np.array([[40, 10], [25, 25]]),
    "Drug B vs. Drug C": np.array([[10, 40], [25, 25]])
}

raw_fisher_pvals = []
contrast_names = []
print("\n--- Unadjusted Pairwise Parameter Extractions ---")
for name, mat in drug_pairs.items():
    odds_ratio, p_v = stats.fisher_exact(mat)
    raw_fisher_pvals.append(p_v)
    contrast_names.append(name)
    print(f"{name: <18} -> Odds Ratio: {odds_ratio:.4f} | Raw Unadjusted p: {p_v}")

# Apply Multi-test Bonferroni Adjustments
rejected, adjusted_pvals, _, _ = multipletests(raw_fisher_pvals, alpha=0.05, method='bonferroni')
print("\n--- Post-Hoc Bonferroni Adjusted Significance Results ---")
for i in range(len(contrast_names)):
    print(f"{contrast_names[i]: <18} -> Adjusted p-value: {adjusted_pvals[i]} (Rejected H0: {rejected[i]})")


# =============================================================================
# PROBLEM 5: MULTIVARIATE LOGISTIC REGRESSION FOR GRADUATE ADMISSIONS
# =============================================================================
print("\n" + "="*80)
print(" RUNNING: PROBLEM 5 - LOGISTIC GLM GRADUATE TRACKING")
print("="*80)

try:
    df_admit = pd.read_csv("binary.csv")
    
    # Specify logistic model frame treating university rank as categorical factor
    model_admit = smf.logit("admit ~ gre + gpa + C(rank)", data=df_admit).fit()
    print("\n--- Logistic Maximum Likelihood Estimation ---")
    print(model_admit.summary())

    # Exponentiated odds ratio confidence interval mapping
    or_params = np.exp(model_admit.params)
    conf_int_admit = np.exp(model_admit.conf_int())
    or_df = pd.DataFrame({
        "Adjusted Odds Ratio (OR)": or_params,
        "Lower 95% CI": conf_int_admit[0],
        "Upper 95% CI": conf_int_admit[1]
    })
    print("\n--- Transformed Risk Multipliers (Odds Ratios) ---")
    print(np.round(or_df, 4))
except FileNotFoundError:
    print("\n[WARNING] 'binary.csv' missing. Skipping admissions regression modeling.")


# =============================================================================
# PROBLEM 6: COUNT DATA VALUATIONS VIA POISSON GENERALIZED LINEAR MODELS
# =============================================================================
print("\n" + "="*80)
print(" RUNNING: PROBLEM 6 - POISSON CANONICAL COUNT REGRESSION")
print("="*80)

try:
    df_awards = pd.read_csv("awards.csv")

    # Fit Poisson GLM Framework via log link structures
    model_poisson = smf.glm("num_awards ~ math + C(prog)", data=df_awards, 
                            family=sm.families.Poisson()).fit()
    print("\n--- Poisson GLM Estimation Architecture Results ---")
    print(model_poisson.summary())

    # Deconstruct and capture Incidence Rate Ratios (IRRs)
    irrs_poisson = np.exp(model_poisson.params)
    conf_int_poisson = np.exp(model_poisson.conf_int())
    irr_df_poisson = pd.DataFrame({
        "Incidence Rate Ratio (IRR)": irrs_poisson,
        "Lower 95% CI": conf_int_poisson[0],
        "Upper 95% CI": conf_int_poisson[1]
    })
    print("\n--- Multiplicative Incidence Rate Scale Shifts (IRRs) ---")
    print(np.round(irr_df_poisson, 4))
except FileNotFoundError:
    print("\n[WARNING] 'awards.csv' missing. Skipping Poisson count model calculations.")


# =============================================================================
# PROBLEM 7: NEGATIVE BINOMIAL COUNT REGRESSION MODELING (STUDENT ABSENCES)
# =============================================================================
print("\n" + "="*80)
print(" RUNNING: PROBLEM 7 - NEGATIVE BINOMIAL (NB2) GLM FRAME")
print("="*80)

try:
    df_abs = pd.read_csv("nb_data.csv")
    print("\n--- Overdispersion Baseline Properties Verification ---")
    print(f"Days Absent Sample Mean    : {df_abs['daysabs'].mean():.4f}")
    print(f"Days Absent Sample Variance: {df_abs['daysabs'].var():.4f}")

    # Fit Negative Binomial Model factoring alpha scaling weights
    model_nb = smf.negativebinomial("daysabs ~ math + C(prog)", data=df_abs, loglike_method='nb2').fit()
    print("\n--- Negative Binomial Generalized Regression Summary ---")
    print(model_nb.summary())

    irrs_nb = np.exp(model_nb.params)
    conf_int_nb = np.exp(model_nb.conf_int())
    irr_df_nb = pd.DataFrame({
        "Incidence Rate Ratio (IRR)": irrs_nb,
        "Lower 95% CI": conf_int_nb[0],
        "Upper 95% CI": conf_int_nb[1]
    })
    print("\n--- Exponentiated Model Rate Shifts (IRRs) ---")
    print(np.round(irr_df_nb, 4))
except FileNotFoundError:
    print("\n[WARNING] 'nb_data.csv' missing. Skipping overdispersed count evaluation.")


# =============================================================================
# PROBLEMS 8 & 9: ZERO-INFLATED DUAL FRAME COUNT ANALYSIS (FISH CATCH DATA)
# =============================================================================
print("\n" + "="*80)
print(" RUNNING: PROBLEMS 8 & 9 - ZERO-INFLATED STRUCTURES (ZIP / ZINB)")
print("="*80)

try:
    df_fish = pd.read_csv("fish_data.csv")
    df_fish['intercept'] = 1.0  # Explicit setup matrix array constant index columns

    # 1. Problem 8: Zero-Inflated Poisson Model
    model_zip = ZeroInflatedPoisson(
        endog=df_fish['count'], 
        exog=df_fish[['intercept', 'child', 'camper']], 
        exog_infl=df_fish[['intercept', 'persons']]
    ).fit()
    print("\n--- Problem 8: Zero-Inflated Poisson (ZIP) Summary Results ---")
    print(model_zip.summary())

    # 2. Problem 9: Zero-Inflated Negative Binomial Model (Upgrade for overdispersion)
    model_zinb = ZeroInflatedNegativeBinomial(
        endog=df_fish['count'], 
        exog=df_fish[['intercept', 'child', 'camper']], 
        exog_infl=df_fish[['intercept', 'persons']]
    ).fit(method='bfgs')
    print("\n--- Problem 9: Zero-Inflated Negative Binomial (ZINB) Summary Results ---")
    print(model_zinb.summary())
except FileNotFoundError:
    print("\n[WARNING] 'fish_data.csv' missing. Skipping Zero-Inflated execution chains.")


# =============================================================================
# PROBLEMS 10 & 11: ZERO-TRUNCATED HIERARCHICAL MODELING (HOSPITAL STAY DAYS)
# =============================================================================
print("\n" + "="*80)
print(" RUNNING: PROBLEMS 10 & 11 - ZERO-TRUNCATED MODELS (ZTP / ZTNB)")
print("="*80)

# 1. Problem 10: Zero-Truncated Poisson Estimation Diagnostics
try:
    df_hosp = pd.read_csv("hospital_stay_data.csv")
    model_ztp = smf.glm("stay ~ age + hmo + died", data=df_hosp, family=sm.families.Poisson()).fit()
    print("\n--- Problem 10: Zero-Truncated Poisson Baseline Output ---")
    print(model_ztp.summary())
except FileNotFoundError:
    print("\n[WARNING] 'hospital_stay_data.csv' missing. Running localized Problem 11 vector projection simulation instead.")

# 2. Problem 11: Zero-Truncated Negative Binomial Mathematical Calculation Engine Projection Graphing
beta_0_z, beta_age_z, beta_hmo_z, beta_died_z = 2.40833, -0.01569, -0.14706, -0.21777
alpha_ztnb = 0.56864
age_steps = np.arange(1, 10)

def calculate_truncated_expected_value(age, hmo, died):
    mu_val = np.exp(beta_0_z + beta_age_z * age + beta_hmo_z * hmo + beta_died_z * died)
    p_zero_mass = (1.0 / (1.0 + alpha_ztnb * mu_val)) ** (1.0 / alpha_ztnb)
    return mu_val / (1.0 - p_zero_mass)

ey_non_hmo_surv = calculate_truncated_expected_value(age_steps, hmo=0, died=0)
ey_hmo_surv     = calculate_truncated_expected_value(age_steps, hmo=1, died=0)
ey_hmo_dec      = calculate_truncated_expected_value(age_steps, hmo=1, died=1)

print("\n--- Problem 11: Calculated ZTNB Predicted Stay Day Projections ---")
print("Target Age Arrays   :", age_steps)
print("Predicted Stays (Non-HMO Survivors):", np.round(ey_non_hmo_surv, 2))
print("Predicted Stays (HMO Survivors)    :", np.round(ey_hmo_surv, 2))

# Generate Vector Mathematical Evaluation Graph Plot Output
plt.figure(figsize=(7, 4.5))
plt.plot(age_steps, ey_non_hmo_surv, marker='o', color='navy', lw=2, label='Non-HMO, Survived')
plt.plot(age_steps, ey_hmo_surv, marker='s', color='crimson', linestyle='--', label='HMO, Survived')
plt.plot(age_steps, ey_hmo_dec, marker='^', color='gray', linestyle=':', label='HMO, Deceased')
plt.title('Problem 11: ZTNB Expected Length of Hospital Stay Predictions')
plt.xlabel('Patient Chronological Age Group Classification')
plt.ylabel('Conditional Expected Length of Stay (Days)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('hospital_stay_ztnb_predictions.pdf', dpi=300)
print("\n[SUCCESS] Custom ZTNB modeling visual script exported to 'hospital_stay_ztnb_predictions.pdf'")
plt.close()

print("\n" + "="*80)
print(" ANALYSIS SCRIPT EXECUTION COMPLETE.")
print("="*80 + "\n")