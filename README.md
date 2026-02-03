# 🎓 Chandrakant Sir Finance Dashboard

**Professional Financial Analytics & Reporting System**


The **Chandrakant Sir Finance Dashboard** is a comprehensive tool designed to analyze bank statements, visualize financial health, and provide actionable insights. Built with Python and Streamlit, it transforms raw Excel transaction logs into interactive dashboards, helping users track income, expenses, debts, and future trends.

### 🌐 [Live Demo](https://financial-analytics-dashboard-wa2euufta2xcqgahmxosrh.streamlit.app/)


## 🚀 Key Features

### 📊 1. Analytics Overview
- **High-Level Metrics**: Instant view of total admissions, installments collected, fee revenue, and net cash flow.
- **Revenue Analysis**: Compare Fee Income vs. Personal Transfers interactively.
- **Trend Visualization**: Track Net Flow trends over time with dynamic line charts.
- **🔮 AI Future Projection**: Predicts net balance for the next 3 months based on historical data trends.

### 💰 2. Financial Deep Dive
- **Expense Breakdown**: Categorize expenses (e.g., Personal vs. Business) with pie charts and detailed tables.
- **Transaction Size Analysis**: Understand transaction frequency and volume across different amount ranges (e.g., 0-100, 1k-10k).

### 🧠 3. Relationship Intelligence
- **Personal Ledger System**: Tracks "Who owes whom?" by calculating total received, total paid, and net balance for key contacts.
- **🕸️ Network Graph**: Visualizes the flow of money between the institute and individuals using an interactive network graph (Green = Inflow, Red = Outflow).
- **History Drill-Down**: Select any person to view their full transaction history with the institute.

### 🚨 4. Leak Detector
Identifies potential financial drains:
- **☕ Frequent Small Leaks**: recurring small expenses that add up.
- **🐳 Vendor Dependency**: Highlights major vendors consuming a large chunk of outflow.
- **📈 Spending Spikes**: Detects statistically significant unusual high-value transactions.
- **🔄 Recurring Subscriptions**: Flags payments of exact amounts that occur frequently.

### 📋 5. Raw Data Explorer
- **Advanced Search**: Filter transactions by description, amount range, and category.
- **Data Export**: Download detailed monthly summaries as CSV files.

## 🛠️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/)
- **Visualization**: [Plotly Express](https://plotly.com/python/plotly-express/) & [Graph Objects](https://plotly.com/python/graph-objects/)
- **Graph Analysis**: [NetworkX](https://networkx.org/)

## 📥 Installation

1.  **Clone the repository** (or download usage files):
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install Dependencies**:
    Ensure you have Python installed. Then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**:
    ```bash
    streamlit run app.py
    ```

## 💡 Usage
1.  Launch the app using the command above.
2.  In the **Sidebar**, upload your bank statement (Must be an `.xlsx` file).
    *   *Note: The system expects specific column headers. Ensure your Excel file matches the format expected by `data_processor.py`.*
3.  Once loaded, use the sidebar to filter data by **Year**.
4.  Navigate through the tabs to explore different insights.

## 📂 Project Structure
- `app.py`: The main entry point containing the Streamlit UI and dashboard layout.
- `modules/`:
    - `data_processor.py`: Handles loading and cleaning of raw Excel data.
    - `finance_logic.py`: Contains core business logic for classification, sorting, graph generation, and predictions.
    - `config.py`: Configuration constants (e.g., thresholds for leak detection).