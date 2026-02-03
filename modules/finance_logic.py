
import pandas as pd
import os
from . import config
import networkx as nx
import numpy as np

# ---- MEMORY MGMT ----
def load_category_memory():
    if os.path.exists(config.MEMORY_FILE):
        try:
            return pd.read_csv(config.MEMORY_FILE)
        except:
            return pd.DataFrame(columns=["Description", "Category"])
    return pd.DataFrame(columns=["Description", "Category"])

def get_memory_dict(memory_df):
    # Convert dataframe to dict for faster lookup: {Description: Category}
    # Note: Description lookup in original code was 'substring' based (row["Description"] in desc).
    # If we want O(1), we need exact match. But the original was substring.
    # We will optimize the substring search or keep it as is if list is small.
    # To follow the prompt's "dictionary" suggestion strictly for O(1), we'd need exact keys.
    # But financial descriptions vary. Let's stick to the list for substring but optimize checking.
    # We can't really do O(1) for substring search easily without a more complex trie or Aho-Corasick.
    # We will stick to the existing logic but encapsulated cleanly.
    return memory_df

def classify_transfer(description, memory_df):
    if pd.isna(description):
        return "Other"
    
    desc = str(description).lower()
    
    # A. Check Memory (Priority 1)
    # Optimization: iterate over memory only if memory exists
    if not memory_df.empty:
        for _, row in memory_df.iterrows():
            if str(row["Description"]).lower() in desc:
                return row["Category"]
    
    # B. Fallback Rules (Priority 2)
    if any(name in desc for name in config.PERSONAL_KEYWORDS):
        return "Personal"
    if any(word in desc for word in config.CASH_KEYWORDS):
        return "Cash Withdrawal"
    if any(word in desc for word in config.BUSINESS_KEYWORDS):
        return "Business"
        
    return "Other"

def apply_classification(df, desc_col):
    memory = load_category_memory()
    # Pre-computation or optimization could happen here
    df["Transfer_Type"] = df[desc_col].apply(lambda x: classify_transfer(x, memory))
    
    # Helper for Personal boolean
    df["Personal"] = df["Transfer_Type"] == "Personal"
    return df

def detect_installments(df):
    # Advanced Split Installment Logic
    # Group daily non-admission credits to find installments sum > config.INSTALLMENT_AMOUNT
    
    # Need to identify Admissions first if not already done
    if "Admission" not in df.columns:
        df["Admission"] = df["Credit"].between(9999, 10001)

    potential_inst = df[
        (~df["Admission"]) & 
        (~df["Personal"]) & 
        (df["Credit"] > 0)
    ].copy()

    daily_credit = potential_inst.groupby(["Month", "Date"])["Credit"].sum().reset_index()
    daily_credit["Inst_Count"] = (daily_credit["Credit"] // config.INSTALLMENT_AMOUNT).astype(int)
    daily_credit["Inst_Val"] = daily_credit["Inst_Count"] * config.INSTALLMENT_AMOUNT
    
    return daily_credit

def calculate_monthly_summary(df, daily_credit):
    # Aggregations for Charts/Tables
    monthly_inst_counts = daily_credit.groupby("Month")["Inst_Count"].sum()
    monthly_inst_value = daily_credit.groupby("Month")["Inst_Val"].sum()

    monthly_adm_count = df[df["Admission"]].groupby("Month").size()
    monthly_adm_val = df[df["Admission"]].groupby("Month")["Credit"].sum()
    
    monthly_personal = df[df["Personal"]].groupby("Month")["Credit"].sum()
    monthly_expense = df.groupby("Month")["Debit"].sum()
    
    total_credit = df.groupby("Month")["Credit"].sum()
    
    # Calculate Other Income
    monthly_other = total_credit - (
        monthly_adm_val.reindex(total_credit.index, fill_value=0) + 
        monthly_inst_value.reindex(total_credit.index, fill_value=0) + 
        monthly_personal.reindex(total_credit.index, fill_value=0)
    )
    
    monthly_net = total_credit - monthly_expense

    # Summary DataFrame
    summary_df = pd.DataFrame({
        "Admissions": monthly_adm_count,
        "Installments": monthly_inst_counts,
        "Fee Income": (monthly_adm_val.reindex(total_credit.index, fill_value=0) + monthly_inst_value.reindex(total_credit.index, fill_value=0)),
        "Personal Transfers": monthly_personal,
        "Other Income": monthly_other,
        "Total Expense": monthly_expense,
        "Net Flow": monthly_net
    }).fillna(0).sort_index()
    
    return summary_df

def calculate_financial_health_score(df, summary_df):
    # Global Stats
    total_balance = summary_df["Net Flow"].sum()

    # 1. Prepare Data
    monthly_health = df.groupby("Month")[["Credit", "Debit"]].sum()
    monthly_health["Savings"] = monthly_health["Credit"] - monthly_health["Debit"]
    
    # 2. Factor 1: Savings Rate (30%)
    avg_income = monthly_health["Credit"].mean()
    avg_savings = monthly_health["Savings"].mean()
    savings_rate = (avg_savings / avg_income) if avg_income > 0 else 0
    score_savings = min(max(savings_rate * 100, 0), 100)
    
    # 3. Factor 2: Expense Stability (20%)
    exp_volatility = monthly_health["Debit"].std()
    avg_expense = monthly_health["Debit"].mean()
    if avg_expense > 0:
        cv_exp = exp_volatility / avg_expense
        score_stability = max(0, 100 - (cv_exp * 100))
    else:
        score_stability = 100 

    # 4. Factor 3: Cash Runway (20%)
    if "Balance" in df.columns:
        current_balance_actual = df.iloc[-1]["Balance"] 
    else:
        current_balance_actual = total_balance 
        
    monthly_burn = avg_expense
    if monthly_burn > 0:
        runway_months = current_balance_actual / monthly_burn
    else:
        runway_months = 12 
    score_runway = min((runway_months / 6) * 100, 100)

    # 5. Factor 4: Income Consistency (15%)
    inc_volatility = monthly_health["Credit"].std()
    if avg_income > 0:
        cv_inc = inc_volatility / avg_income
        score_income = max(0, 100 - (cv_inc * 100))
    else:
        score_income = 0

    # 6. Factor 5: Leak Risk (15%)
    small_leak_sum = df[(df["Debit"] > 0) & (df["Debit"] <= config.SMALL_LEAK_THRESHOLD)]["Debit"].sum()
    total_debit_sum = df["Debit"].sum()
    if total_debit_sum > 0:
        leak_ratio = small_leak_sum / total_debit_sum
        score_leak = max(0, 100 - (leak_ratio * 200))
    else:
        score_leak = 100

    # FINAL WEIGHTED SCORE
    final_health_score = (
        score_savings * 0.30 +
        score_stability * 0.20 +
        score_runway * 0.20 +
        score_income * 0.15 +
        score_leak * 0.15
    )
    
    return final_health_score

def get_health_status(score):
    if score >= 80: return "Excellent", "green"
    elif score >= 60: return "Healthy", "blue"
    elif score >= 40: return "Warning", "orange"
    return "Critical", "red"

def build_ledger(df, desc_col):
    def extract_person(desc):
        if pd.isna(desc): return None
        desc = str(desc).lower()
        for name in config.KNOWN_PEOPLE:
            if name in desc:
                return name.title()
        return None

    ledger_df = df.copy()
    ledger_df["Person"] = ledger_df[desc_col].apply(extract_person)
    person_txns = ledger_df[ledger_df["Person"].notna()].copy()
    
    if person_txns.empty:
        return pd.DataFrame(), pd.DataFrame()

    ledger = person_txns.groupby("Person").agg(
        Total_Received=("Credit", "sum"),
        Total_Paid=("Debit", "sum"),
        Transactions=("Person", "count")
    ).reset_index()
    
    ledger["Net_Balance"] = ledger["Total_Received"] - ledger["Total_Paid"]
    ledger["Status"] = ledger["Net_Balance"].apply(
        lambda x: "🟢 They Paid You More" if x >= 0 else "🔴 You Paid Them More"
    )
    ledger = ledger.sort_values("Net_Balance", ascending=False)
    
    return ledger, person_txns

def generate_network_graph(df, desc_col):
    """
    Creates a NetworkX graph representing the transaction relationships.
    Nodes: "Institute" (Central), and other entities.
    Edges: Directed flow of money.
    """
    G = nx.DiGraph()
    center_node = "Institute"
    G.add_node(center_node, node_type="center")

    # Reuse extract_person logic or create general entity extraction if we want to graph EVERYONE
    # For now, let's graph the Known People since that's clean
    ledger, _ = build_ledger(df, desc_col)
    
    if ledger.empty:
        return None

    for _, row in ledger.iterrows():
        person = row["Person"]
        received = row["Total_Received"]
        paid = row["Total_Paid"]
        
        G.add_node(person, node_type="person")
        
        # Edge: Person -> Institute (Money Received)
        if received > 0:
            G.add_edge(person, center_node, weight=received, color="green")
        
        # Edge: Institute -> Person (Money Paid)
        if paid > 0:
            G.add_edge(center_node, person, weight=paid, color="red")
            
    return G

def predict_future_balance(summary_df):
    """
    Predicts the Net Flow (Balance) for the next 3 months using Linear Regression.
    Returns a dataframe of future predictions.
    """
    # Prepare data for regression
    # X axis: Integer representation of months (0, 1, 2...)
    # Y axis: Net Flow
    
    if summary_df.empty or len(summary_df) < 2:
        return None
        
    y_values = summary_df["Net Flow"].values
    x_values = np.arange(len(y_values))
    
    # Fit linear model (degree 1)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    
    # Predict next 3 months
    future_x = np.arange(len(y_values), len(y_values) + 3)
    future_y = slope * future_x + intercept
    
    # Create future period labels
    last_period = summary_df.index[-1]
    
    future_labels = []
    current = pd.Period(last_period, freq='M')
    for _ in range(3):
        current = current + 1
        future_labels.append(str(current))
        
    future_df = pd.DataFrame({
        "Month": future_labels,
        "Predicted Net Flow": future_y
    }).set_index("Month")
    
    return future_df
