
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from modules import data_processor, finance_logic, config

# ---- CONFIGURATION ----
st.set_page_config(layout="wide", page_title="Chandrakant Sir Finance Dashboard", page_icon="🎓")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .stApp {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ---- TITLE & SIDEBAR ----
st.title("🎓 Chandrakant Sir Finance Dashboard")
st.markdown("Professional financial analytics and reporting system.")

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload Bank Statement (XLSX)", type=["xlsx"])
    
    st.markdown("---")
    st.caption("Filters will appear here after data load.")

if uploaded_file:
    # ---- 1. DATA LOADING & CLEANING ----
    df_raw, date_col_name, original_desc_col = data_processor.load_and_clean_data(uploaded_file)
    
    if df_raw is None:
        st.stop()

    desc_col = "Clean_Description"

    # ---- 2. SIDEBAR FILTERS ----
    with st.sidebar:
        years = sorted(df_raw["Year"].unique(), reverse=True)
        selected_years = st.multiselect("Select Years", years, default=years)
        
        if not selected_years:
            st.warning("Please select at least one year.")
            st.stop()
            
        df = df_raw[df_raw["Year"].isin(selected_years)].copy()

    # ---- 3. ADVANCED LOGIC IMPLEMENTATION ----
    
    # Apply Classification Logic
    df = finance_logic.apply_classification(df, desc_col)
    
    # Detect Installments logic
    daily_credit = finance_logic.detect_installments(df)
    
    # Calculate Summaries
    summary_df = finance_logic.calculate_monthly_summary(df, daily_credit)
    
    # Global Stats
    total_admissions = int(summary_df["Admissions"].sum())
    total_installments = int(summary_df["Installments"].sum())
#    total_fee_income = summary_df["Fee Income"].sum() # Was used? yes.
    total_fee_income = summary_df["Fee Income"].sum()
    total_balance = summary_df["Net Flow"].sum()

    # Health Score
    final_health_score = finance_logic.calculate_financial_health_score(df, summary_df)
    health_status, health_color = finance_logic.get_health_status(final_health_score)

    # ---- 4. UI LAYOUT ----

    # Top KPI Banner
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Students Admitted", total_admissions, delta="Count", delta_color="off")
    k2.metric("Installments Collected", total_installments, delta="Count", delta_color="off")
    k3.metric("Total Fee Revenue", f"₹{total_fee_income:,.0f}")
    k4.metric("Net Cash Flow", f"₹{total_balance:,.0f}", 
              delta="Positive" if total_balance > 0 else "Negative",
              delta_color="normal" if total_balance > 0 else "inverse")

    st.markdown("---")

    # Tabs for Organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Analytics Overview", "💰 Financial Deep Dive", "🧠 Relationship Intelligence", "🚨 Leak Detector", "📋 Raw Data"])

    # ... [Keeping visualization logic here for now] ...
    
    with tab1:
        # Check for highest expense day
        max_debit_idx = df["Debit"].idxmax()
        max_row = df.loc[max_debit_idx]
        
        st.info(f"💡 **Insight:** The highest expense occurred on **{max_row[date_col_name].strftime('%d %b %Y')}** with an amount of **₹{max_row['Debit']:,.0f}** ({max_row[desc_col]}).")

        col1, col2 = st.columns(2)
        
        with col1:
            # Interactive Bar Chart: Fee Income vs Personal
            fig_income = px.bar(
                summary_df, 
                x=summary_df.index, 
                y=["Fee Income", "Personal Transfers"], 
                title="Revenue Sources: Fee vs Personal",
                labels={"value": "Amount (₹)", "Month": "Month", "variable": "Source"},
                barmode='group',
                color_discrete_sequence=["#2ecc71", "#3498db"] # Green and Blue
            )
            st.plotly_chart(fig_income, width="stretch")

        with col2:
            # Interactive Line Chart: Net Flow
            fig_flow = go.Figure()
            fig_flow.add_trace(go.Scatter(
                x=summary_df.index, 
                y=summary_df["Net Flow"],
                mode='lines+markers',
                name='Net Flow',
                line=dict(color='#e74c3c', width=3),
                fill='tozeroy' # Fill area
            ))
            fig_flow.update_layout(title="Net Financial Flow Trend", xaxis_title="Month", yaxis_title="Net Amount (₹)")
            st.plotly_chart(fig_flow, width="stretch")

        # ---- FUTURE PREDICTION (NEW) ----
        st.markdown("---")
        st.subheader("🔮 AI Future Projection")
        st.caption("Based on your recent trend, here is the estimated Net Balance for the next 3 months.")
        
        future_df = finance_logic.predict_future_balance(summary_df)
        if future_df is not None:
            fig_pred = go.Figure()
            
            # Historical
            fig_pred.add_trace(go.Scatter(
                x=summary_df.index, y=summary_df["Net Flow"],
                mode='lines+markers', name='Actual History',
                line=dict(color='#3498db')
            ))
            
            # Prediction
            # Connect the last actual point to the first predicted point for continuity
            last_actual_val = summary_df["Net Flow"].iloc[-1]
            last_actual_date = summary_df.index[-1]
            
            pred_x = [last_actual_date] + list(future_df.index)
            pred_y = [last_actual_val] + list(future_df["Predicted Net Flow"])
            
            fig_pred.add_trace(go.Scatter(
                x=pred_x, y=pred_y,
                mode='lines+markers', name='AI Prediction',
                line=dict(color='#e67e22', dash='dash')
            ))
            
            st.plotly_chart(fig_pred, width="stretch")
        else:
            st.info("Not enough data points to generate a reliable future prediction.")

    with tab2:
        c1, c2 = st.columns(2)
        
        with c1:
            # Expense Trend
            fig_exp = px.bar(
                summary_df,
                x=summary_df.index,
                y="Total Expense",
                title="Monthly Expenses",
                color="Total Expense",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_exp, width="stretch", key="chart_exp")
            
        with c2:
            # CLASSIFICATION ANALYSIS
            st.subheader("Money Outflow Classification")
            
            # Filter outgoing only for this chart
            outgoing_only = df[df["Debit"] > 0]
            type_counts = outgoing_only.groupby("Transfer_Type")["Debit"].sum().reset_index()
            
            fig_type = px.pie(
                type_counts, 
                values="Debit", 
                names="Transfer_Type", 
                title="Expenses by Category (Personal vs Business)",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_type, width="stretch", key="chart_type")

            # Breakdown Table
            st.caption("Expense Breakdown by Category")
            st.dataframe(
                type_counts.style.format({"Debit": "₹{:,.0f}"}),
                width="stretch"
            )
            
        st.markdown("---")
        
        # MEANINGFUL DISTRIBUTIONS
        st.subheader("📊 Transaction Size Distribution")
        
        # 1. Prepare Data
        dist_df = df.copy()
        dist_df["Amount"] = dist_df["Debit"] + dist_df["Credit"]
        dist_df = dist_df[dist_df["Amount"] > 0].copy()
        
        # 2. Define Buckets
        bins = [0, 100, 500, 1000, 10000, 50000, 100000, float("inf")]
        labels = ["0-100", "100-500", "500-1k", "1k-10k", "10k-50k", "50k-100k", "100k+"]
        
        dist_df["Range"] = pd.cut(dist_df["Amount"], bins=bins, labels=labels)
        
        # 3. Aggregation
        range_summary = dist_df.groupby("Range", observed=False).agg(
            Txn_Count=("Amount", "count"),
            Total_Value=("Amount", "sum"),
            Avg_Value=("Amount", "mean")
        ).reset_index()
        
        # 4. Visualizations
        t1, t2 = st.columns(2)
        
        with t1:
            fig_count = px.bar(
                range_summary, 
                x="Range", 
                y="Txn_Count",
                title="1️⃣ Frequency: How many transactions?",
                labels={"Txn_Count": "Number of Transactions", "Range": "Amount Range (₹)"},
                color="Txn_Count",
                color_continuous_scale="Viridis"
            )
            fig_count.update_layout(xaxis_title=None)
            st.plotly_chart(fig_count, width="stretch", key="chart_dist_count")
            
        with t2:
            fig_val = px.bar(
                range_summary, 
                x="Range", 
                y="Total_Value",
                title="2️⃣ Volume: Where is the money?",
                labels={"Total_Value": "Total Value (₹)", "Range": "Amount Range (₹)"},
                color="Total_Value",
                color_continuous_scale="Magma"
            )
            fig_val.update_layout(xaxis_title=None)
            st.plotly_chart(fig_val, width="stretch", key="chart_dist_val")
            
        st.dataframe(
            range_summary.style.format({
                "Total_Value": "₹{:,.0f}",
                "Avg_Value": "₹{:,.0f}"
            }).background_gradient(subset=["Total_Value"], cmap="magma"),
            width="stretch"
        )

    with tab3:
        st.subheader("📒 Personal Ledger System")
        st.caption("Track net balances with your key contacts. Who owes whom?")

        ledger, person_txns = finance_logic.build_ledger(df, desc_col)
        
        if not ledger.empty:
            
            # A. Top Level Metrics
            l1, l2, l3 = st.columns(3)
            # Total Volume = Sum of all credits and debits involving these people
            total_volume = ledger['Total_Received'].sum() + ledger['Total_Paid'].sum()
            
            l1.metric("Total Tracked Volume", f"₹{total_volume:,.0f}")
            l2.metric("Total Received (Inflow)", f"₹{ledger['Total_Received'].sum():,.0f}", delta="Credit", delta_color="normal")
            l3.metric("Total Paid (Outflow)", f"₹{ledger['Total_Paid'].sum():,.0f}", delta="Debit", delta_color="inverse")
            
            st.markdown("---")
            
            # B. The Main Ledger Pivot
            c_led1, c_led2 = st.columns([2, 1])
            
            with c_led1:
                st.subheader("Balance Sheet")
                st.dataframe(
                    ledger.style.format({
                        "Total_Received": "₹{:,.0f}", 
                        "Total_Paid": "₹{:,.0f}", 
                        "Net_Balance": "₹{:,.0f}"
                    }).background_gradient(subset=["Net_Balance"], cmap="RdYlGn", vmin=-50000, vmax=50000),
                    width="stretch"
                )
                
            with c_led2:
                st.subheader("Net Balance Viz")
                fig_bal = px.bar(ledger, x="Net_Balance", y="Person", orientation='h',
                                 title="Net Balance (Received - Paid)",
                                 color="Net_Balance", 
                                 color_continuous_scale="RdYlGn",
                                 labels={"Net_Balance": "Net Amount (₹)"})
                
                fig_bal.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
                st.plotly_chart(fig_bal, width="stretch", key="chart_balance")
            
            # ---- NETWORK GRAPH (NEW) ----
            st.markdown("---")
            st.subheader("🕸️ Transaction Network Graph")
            st.caption("Visualizing money flow. Green = Inflow to Institute, Red = Outflow from Institute.")
            
            graph = finance_logic.generate_network_graph(df, desc_col)
            if graph:
                # Spring layout
                pos = nx.spring_layout(graph, seed=42)
                
                # Extract edges
                edge_x = []
                edge_y = []
                edge_colors = []
                
                for edge in graph.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_colors.append(edge[2]['color'])

                # Middle node trace (Institute)
                node_x = []
                node_y = []
                node_text = []
                node_colors = []
                
                for node in graph.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                    if node == "Institute":
                        node_colors.append("#f1c40f") # Gold
                    else:
                        node_colors.append("#bdc3c7") # Grey

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="top center",
                    marker=dict(
                        showscale=False,
                        color=node_colors,
                        size=20,
                        line_width=2))
                
                fig_net = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                title='Transaction Relationships',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=0,l=0,r=0,t=0),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                )
                st.plotly_chart(fig_net, width="stretch")
            
            st.markdown("---")
            
            # C. Drill Down History
            st.subheader("🕵️ Person History Drill-Down")
            target_person = st.selectbox("Select Person to View History", ledger["Person"].unique())
            
            if target_person:
                history = person_txns[person_txns["Person"] == target_person].reset_index(drop=True)
                
                cols_to_show = [date_col_name, desc_col, original_desc_col, "Debit", "Credit"]
                history = history.sort_values(date_col_name, ascending=False)
                
                p_recv = history["Credit"].sum()
                p_paid = history["Debit"].sum()
                p_net = p_recv - p_paid
                
                ph1, ph2, ph3 = st.columns(3)
                ph1.metric(f"Received from {target_person}", f"₹{p_recv:,.0f}")
                ph2.metric(f"Paid to {target_person}", f"₹{p_paid:,.0f}")
                ph3.metric("Net Balance", f"₹{p_net:,.0f}", 
                           delta="Positive" if p_net >= 0 else "Negative", 
                           delta_color="normal" if p_net >= 0 else "inverse")
                
                st.dataframe(
                    history[cols_to_show].style.format({
                        "Credit": "₹{:,.0f}", 
                        "Debit": "₹{:,.0f}"
                    }, na_rep="-"),
                    width="stretch"
                )
        else:
            st.info("No known persons found in the transaction log.")


    with tab4:
        st.subheader("🚨 Expense Leak Detector")
        st.caption("Identify silent money drains, unusual spikes, and recurring billing patterns.")
        
        # 1. FREQUENT SMALL PAYMENTS
        st.markdown(f"#### 1. ☕ Frequent Small Leaks (<= ₹{config.SMALL_LEAK_THRESHOLD})")
        small_txn = df[(df["Debit"] > 0) & (df["Debit"] <= config.SMALL_LEAK_THRESHOLD)]
        
        if not small_txn.empty:
            small_summary = small_txn.groupby(desc_col).agg(
                Transactions=("Debit", "count"),
                Total_Spent=("Debit", "sum")
            ).sort_values("Transactions", ascending=False).head(10)
            
            c_s1, c_s2 = st.columns([1, 2])
            with c_s1:
                st.dataframe(small_summary, width="stretch")
            with c_s2:
                fig_small = px.bar(small_summary, x=small_summary.index, y="Transactions", 
                                   title="Top Frequent Small Expenses (Count)", color="Total_Spent")
                st.plotly_chart(fig_small, width="stretch", key="chart_small")
        else:
            st.success("No frequent small leaks detected.")

        st.markdown("---")

        # 2. VENDOR DEPENDENCY
        st.markdown("#### 2. 🐳 Vendor Dependency (Major Drains)")
        vendor_spend = df[df["Debit"] > 0].groupby(desc_col)["Debit"].sum().sort_values(ascending=False)
        total_debit = vendor_spend.sum()
        
        if total_debit > 0:
            vendor_ratio = (vendor_spend / total_debit * 100).head(5)
            whales = vendor_ratio[vendor_ratio > config.WHALE_DEPENDENCY_THRESHOLD]
            
            if not whales.empty:
                st.warning(f"⚠️ **High Dependency Detected:** You spend {whales.iloc[0]:.1f}% of your total outflow on **{whales.index[0]}**.")
            
            fig_whale = px.bar(vendor_spend.head(10), orientation='h', 
                               title="Top Vendors by Total Spend", labels={"value": "Amount (₹)", desc_col: "Vendor"})
            fig_whale.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_whale, width="stretch", key="chart_whale")

        st.markdown("---")
        
        # 3. SPENDING SPIKES
        st.markdown("#### 3. 📈 Unusual Spending Spikes")
        debits = df[df["Debit"] > 0]["Debit"]
        if not debits.empty:
            mean_spend = debits.mean()
            std_spend = debits.std()
            threshold = mean_spend + (3 * std_spend)
            
            spikes = df[df["Debit"] > threshold][["Date", desc_col, "Debit", "Transfer_Type"]].sort_values("Debit", ascending=False)
            
            if not spikes.empty:
                st.error(f"Found {len(spikes)} transactions significantly higher than normal (> ₹{threshold:,.0f}).")
                st.dataframe(spikes.style.format({"Debit": "₹{:,.0f}"}), width="stretch")
            else:
                st.success("No unusual spending spikes detected (within 3 deviations).")

        st.markdown("---")

        # 4. RECURRING PAYMENTS
        st.markdown("#### 4. 🔄 Recurring / Subscription Detection")
        recurring = df[df["Debit"] > 0].groupby([desc_col, "Debit"]).size().reset_index(name="Count")
        recurring = recurring[recurring["Count"] >= 3].sort_values("Count", ascending=False)
        
        if not recurring.empty:
            st.info("These payments of the **exact same amount** happen frequently. Check for unwanted subscriptions.")
            st.dataframe(recurring.style.format({"Debit": "₹{:,.0f}"}), width="stretch")
        else:
            st.success("No recurring fixed-amount payments detected.")
            
    with tab5:
        st.subheader("📋 Full Transaction Log")
        st.caption("Search and filter your raw data.")
        
        # ---- FILTERS (NEW) ----
        f1, f2, f3 = st.columns(3)
        with f1:
            search_query = st.text_input("🔍 Search Description", "")
        with f2:
            min_amt, max_amt = st.slider("💰 Amount Range (Debit/Credit)", 
                                         0.0, float(df[["Debit", "Credit"]].max().max()), (0.0, 50000.0))
        with f3:
            type_filter = st.multiselect("🏷️ Category", df["Transfer_Type"].unique(), default=df["Transfer_Type"].unique())
            
        # Apply filters
        mask = (
            (df[desc_col].str.contains(search_query, case=False, na=False)) &
            ((df["Debit"].between(min_amt, max_amt)) | (df["Credit"].between(min_amt, max_amt))) &
            (df["Transfer_Type"].isin(type_filter))
        )
        filtered_df = df[mask]
        
        st.info(f"Showing {len(filtered_df)} transactions.")
        
        # Show specific useful columns
        display_cols = [date_col_name, "Clean_Description", "Debit", "Credit", "Balance", original_desc_col]
        
        # Filter existing columns
        valid_cols = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            filtered_df[valid_cols].sort_values(date_col_name, ascending=False).style.format({
                "Debit": "₹{:,.0f}",
                "Credit": "₹{:,.0f}", 
                "Balance": "₹{:,.0f}"
            }),
            width="stretch",
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.subheader("Month-wise Detailed Summary")
        
        st.dataframe(
            summary_df.style.format("{:,.0f}").background_gradient(subset=["Net Flow"], cmap="RdYlGn"),
            width="stretch"
        )
        
        csv = summary_df.to_csv().encode('utf-8')
        st.download_button(
            "📥 Download Summary Report",
            csv,
            "institute_finance_report.csv",
            "text/csv",
            key='download-csv'
        )

else:
    st.info("👋 Welcome! Please upload your Bank Statement (Excel file) in the sidebar to begin.")
