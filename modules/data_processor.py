
import pandas as pd
import re
import streamlit as st

@st.cache_data
def load_and_clean_data(file):
    """
    Loads the excel file, cleans numeric columns, parses dates, 
    and extracts clean descriptions.
    """
    try:
        df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None, None, None

    # Clean numeric columns
    numeric_cols = ["Debit", "Credit", "Balance"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "")
                .replace("nan", "0")
                .replace("None", "0")
                .replace("", "0")
                .astype(float)
            )
    
    # Date parsing
    # Find a column that looks like 'date'
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if not date_cols:
        st.error("No 'Date' column found in the Excel file.")
        return None, None, None
        
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    df["Month"] = df[date_col].dt.to_period("M").astype(str)
    df["Date"] = df[date_col].dt.date
    df["Year"] = df[date_col].dt.year

    # Name Extraction
    desc_cols = [c for c in df.columns if "desc" in c.lower()]
    if not desc_cols:
        # Fallback if no description column
        original_desc_col = df.columns[1] if len(df.columns) > 1 else "Description"
    else:
        original_desc_col = desc_cols[0]

    df["Clean_Description"] = df[original_desc_col].apply(extract_clean_name)
    
    return df, date_col, original_desc_col

def extract_clean_name(desc):
    if pd.isna(desc): return "Unknown"
    desc_str = str(desc)
    
    # Pattern: UPI/DR/Ref/NAME/Bank
    # Extracts the 4th block in standard UPI strings
    match = re.search(r"UPI/(?:DR|CR)/\d+/([^/]+)/", desc_str)
    if match:
        clean = match.group(1).replace("_", " ").title().strip()
        return clean
    
    return desc_str # Return original if no pattern found
