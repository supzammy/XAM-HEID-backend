"""
Pattern mining utilities using mlxtend (apriori) to discover associations between demographic attributes
and disease presence at the aggregated level. The module exposes functions to transform patient-level
rows into transaction-style data and run apriori + rule extraction.
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from typing import List, Tuple, Dict, Any

from data_loader import apply_rule_of_11


def make_transactions(df: pd.DataFrame, disease: str, groupby: List[str] = ['state','year','income_group', 'age_group', 'sex', 'race_ethnicity']) -> pd.DataFrame:
    """Create a one-hot-encoded transaction table where each row corresponds to a grouped cell (e.g., state-year-income)
    and columns include demographic buckets and disease indicators (e.g., 'income=Low', 'age=65+', 'disease=diabetes').
    This version incorporates the Rule of 11 for privacy.
    """
    df = df.copy()

    # 1. Aggregate data to get counts for cases and population per group
    group_cols = [col for col in groupby if col in df.columns]
    if not group_cols:
        return pd.DataFrame() # Cannot create transactions without grouping

    agg = df.groupby(group_cols).agg(
        population=('patient_id', 'count'),
        cases=(disease, 'sum')
    ).reset_index()

    # 2. Apply the Rule of 11 to suppress small counts
    agg_secure = apply_rule_of_11(agg, case_col='cases', pop_col='population')

    # 3. Filter out the suppressed groups to ensure privacy
    safe_groups = agg_secure[agg_secure['suppressed'] == False].copy()
    if safe_groups.empty:
        return pd.DataFrame()

    # 4. Create transaction "items" from the safe, aggregated data
    # Each row in safe_groups is now a transaction.
    # The items are the demographic values and whether the disease is present.
    items = []
    for _, row in safe_groups.iterrows():
        itemset = []
        # Add demographic characteristics as items
        for col in group_cols:
            if col not in ['state', 'year']: # state/year are for grouping, not features
                 itemset.append(f"{col}={row[col]}")
        
        # Add disease presence as an item.
        # We consider the disease "present" for the group if cases > 0.
        if row['cases'] > 0:
            itemset.append(f"has_{disease}")
        
        items.append(itemset)

    # 5. Convert the list of itemsets into a one-hot encoded DataFrame
    if not items:
        return pd.DataFrame()

    all_items = sorted(list(set(it for itemset in items for it in itemset)))
    
    rows = []
    for itemset in items:
        presence = {item: (1 if item in itemset else 0) for item in all_items}
        rows.append(presence)
        
    tx = pd.DataFrame(rows, columns=all_items)
    return tx


def run_apriori(transactions: pd.DataFrame, min_support: float = 0.05, min_threshold: float = 0.6):
    """Run apriori and extract association rules. Returns frequent itemsets and rules DataFrames.
    `min_threshold` parameter maps to min_confidence for association_rules.
    """
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
    # Sort by lift/confidence for interesting rules
    rules = rules.sort_values(['lift','confidence'], ascending=False)
    return frequent_itemsets, rules


def summarize_rules(rules: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
    """Return a simplified list of top rules with human-readable antecedent/consequent.
    """
    out = []
    for _, row in rules.head(top_n).iterrows():
        out.append({
            'antecedent': tuple(sorted(list(row['antecedents']))),
            'consequent': tuple(sorted(list(row['consequents']))),
            'support': float(row['support']),
            'confidence': float(row['confidence']),
            'lift': float(row['lift'])
        })
    return out
