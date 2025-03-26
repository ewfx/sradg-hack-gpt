import streamlit as st
import pandas as pd
import requests
import logging
import json
from datetime import datetime

# Set light theme and page config
st.set_page_config(
    page_title="AI Agent Powered Reconciliation",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Add custom CSS for light theme
st.markdown("""
<style>
    .stApp {
        background-color: white;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables"""
    if 'process_config' not in st.session_state:
        st.session_state.process_config = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'setup'
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'anomaly_results' not in st.session_state:
        st.session_state.anomaly_results = {}
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

def create_reconciliation_process():
    st.subheader("Create Reconciliation Process")
    with st.form("reconciliation_setup"):
        # Key columns selection
        key_columns_input = st.text_input(
            "Enter Key Columns (comma separated)",
            value="Account,Transaction ID",
            help="Enter column names separated by commas"
        )
        key_columns = [col.strip() for col in key_columns_input.split(",") if col.strip()]
        
        # Derived columns selection
        derived_columns_input = st.text_input(
            "Enter Derived Columns (comma separated)", 
            value="Balance",
            help="Enter column names separated by commas"
        )
        derived_columns = [col.strip() for col in derived_columns_input.split(",") if col.strip()]

        # Break categories selection
        break_categories_input = st.text_input(
            "Enter Break Categories (comma separated)",
            value="AMOUNT_MISMATCH,DATE_MISMATCH",
            help="Enter categories separated by commas"
        )
        break_categories = [cat.strip() for cat in break_categories_input.split(",") if cat.strip()]
        
        # Source configurations
        col1, col2 = st.columns(2)
        with col1:
            source1_name = st.text_input("Source 1 Name", "Bank Statement")
            source1_url = st.text_input("Source 1 URL/Path", "")
        
        with col2:
            source2_name = st.text_input("Source 2 Name", "GL Entry")
            source2_url = st.text_input("Source 2 URL/Path", "")
        
        submit_button = st.form_submit_button("Create Process")
        
        if submit_button:
            st.session_state.process_config = {
                "key_columns": key_columns,
                "derived_columns": derived_columns,
                "break_categories": break_categories,
                "source1": {"name": source1_name, "url": source1_url},
                "source2": {"name": source2_name, "url": source2_url}
            }
            st.success("Reconciliation process created successfully!")
            st.session_state.current_page = 'dashboard'
            st.rerun()

def show_process_config():
    """Display current process configuration"""
    with st.expander("Current Process Configuration", expanded=False):
        config = st.session_state.process_config
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Configuration")
            st.write("**Key Columns:**")
            for col in config["key_columns"]:
                st.write(f"- {col}")
            
            st.write("**Derived Columns:**")
            for col in config["derived_columns"]:
                st.write(f"- {col}")
            
            st.write("**Break Categories:**")
            for cat in config["break_categories"]:
                st.write(f"- {cat}")
        
        with col2:
            st.subheader("Source Systems")
            st.write("**Source 1:**")
            st.write(f"- Name: {config['source1']['name']}")
            st.write(f"- URL/Path: {config['source1']['url']}")
            
            st.write("**Source 2:**")
            st.write(f"- Name: {config['source2']['name']}")
            st.write(f"- URL/Path: {config['source2']['url']}")

def detect_anomalies(record: dict, index: int):
    """Detect anomalies for a single record"""
    try:
        response = requests.post(
            "http://localhost:8000/detect_anomaly/",
            json={
                "key_columns": st.session_state.process_config["key_columns"],
                "break_category": st.session_state.process_config["break_categories"],
                "record": record
            }
        )
        result = response.json()
        st.session_state.anomaly_results[index] = result
        return result
    except Exception as e:
        st.error(f"Error detecting anomalies: {str(e)}")
        return None

def approve_fix(record: dict, feedback: str, index: int):
    """Approve and execute fix for a record"""
    try:
        response = requests.post(
            "http://localhost:8000/approve_fix/",
            json={"record": record, "approval": True, "feedback": feedback}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error approving fix: {str(e)}")
        return None

def show_analysis_results(result, index):
    """Display analysis results in a structured format"""
    # Remove the detect anomaly button after results are shown
    st.empty()  # Clear the previous button
    
    # Create containers for different sections
    anomaly_container = st.container()
    fix_container = st.container()
    action_container = st.container()
    
    # Anomaly Detection Section
    with anomaly_container:
        st.subheader("Anomaly Detection")
        st.write("**Anomaly Detected:**", "Yes" if result.get('final_assessment', {}).get('is_anomaly', True) else "No")
        
        # Safely get the first anomaly's reasoning if available
        anomalies = result.get('llm_analysis', {}).get('anomalies', [])
        if anomalies:
            st.write("**Comments:**", anomalies[0].get('reasoning', 'No comments available'))
        else:
            st.write("**Comments:**", "No anomalies detected")
            
        st.write("**Impact:**", result.get('fix_suggestion', {}).get('risk_assessment', 'No impact assessment available'))
    
    # Fix Details Section
    with fix_container:
        st.subheader("Fix Details")
        st.write("**Cause:**", result.get('fix_suggestion', {}).get('root_cause', 'No cause identified'))
        st.write("**Fix Suggestion:**", result.get('fix_suggestion', {}).get('recommended_fix', 'No fix suggested'))
        st.write("**Implementation Steps:**")
        steps = result.get('fix_suggestion', {}).get('implementation_steps', ['No steps available'])
        for step in steps:
            st.write(f"- {step}")
    
    # Fix Action Section
    with action_container:
        st.subheader("Fix Actions")
        feedback = st.text_area("Feedback (optional):", key=f"feedback_{index}")
        
        # Create a row for buttons using a container
        button_container = st.container()
        with button_container:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Approve Fix", key=f"approve_{index}", type="primary"):
                    with st.spinner("Executing fix..."):
                        fix_result = approve_fix(result.get('record', {}), feedback, index)
                        if fix_result:
                            st.success("Fix executed successfully!")
                            # Show simulation interface
                            show_fix_simulation(fix_result, result)
            
            with col2:
                if st.button("❌ Reject Fix", key=f"reject_{index}", type="secondary"):
                    st.warning("Fix rejected")
                    # Log the rejection with feedback
                    logger.info(f"Fix rejected for record {index} with feedback: {feedback}")

def show_fix_simulation(fix_result, original_result):
    """Display simulation of fix execution in external systems"""
    st.subheader("Fix Execution Simulation")
    
    # Create tabs for different systems
    system_tabs = st.tabs(["Source System 1", "Source System 2", "JIRA Ticket", "Email Notification"])
    
    with system_tabs[0]:
        st.write("**Source System 1 Update**")
        st.info("Simulating update in primary system...")
        # Show simulated changes using containers
        before_container = st.container()
        after_container = st.container()
        
        with before_container:
            st.write("**Before Fix:**")
            st.json(original_result.get('record', {}))
            
        with after_container:
            st.write("**After Fix:**")
            st.json(fix_result.get('result', {}).get('updated_record', {}))
    
    with system_tabs[1]:
        st.write("**Source System 2 Update**")
        st.info("Synchronizing with secondary system...")
        # Show sync status
        st.success("✓ Changes synchronized successfully")
        
    with system_tabs[2]:
        st.write("**JIRA Ticket**")
        ticket_info = {
            "Ticket ID": f"RECON-{hash(str(original_result)) % 1000}",
            "Status": "Resolved",
            "Summary": "Reconciliation Fix Implementation",
            "Description": original_result.get('fix_suggestion', {}).get('recommended_fix', ''),
            "Resolution": "Fix applied successfully"
        }
        st.json(ticket_info)
        
    with system_tabs[3]:
        st.write("**Email Notification**")
        email_preview = {
            "To": "stakeholders@company.com",
            "Subject": f"Reconciliation Fix Completed - {ticket_info['Ticket ID']}",
            "Body": f"""Fix has been successfully implemented.
            
Original Issue: {original_result.get('fix_suggestion', {}).get('root_cause', '')}
Resolution: {original_result.get('fix_suggestion', {}).get('recommended_fix', '')}
Status: Completed
            
Please review the changes and confirm."""
        }
        st.json(email_preview)

def calculate_column_differences(df, key_columns):
    """Calculate differences between numeric columns"""
    differences = {}
    for i in range(len(key_columns)):
        for j in range(i + 1, len(key_columns)):
            col1, col2 = key_columns[i], key_columns[j]
            if col1 in df.columns and col2 in df.columns:
                if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                    sum1, sum2 = df[col1].sum(), df[col2].sum()
                    if sum2 != 0:  # Avoid division by zero
                        diff_percent = ((sum1 - sum2) / sum2) * 100
                        differences[f"{col1}_vs_{col2}"] = {
                            "sum1": sum1,
                            "sum2": sum2,
                            "difference": diff_percent
                        }
    return differences

def show_dashboard():
    """Display the reconciliation dashboard"""
    st.subheader("Reconciliation Dashboard")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload Reconciliation Data",
        type=["xlsx", "csv", "xls"],
        key="file_uploader"
    )

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('xlsx', 'xls')) else pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return

        # Add a download button for the updated data at the top
        if st.button("Download Updated Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="reconciliation_results.csv",
                mime="text/csv"
            )

        # Calculate metrics and show summary
        total_records = len(df)
        processed_records = len(st.session_state.anomaly_results)
        pending_records = total_records - processed_records

        # Display summary metrics
        st.subheader("Summary Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", total_records)
        with col2:
            st.metric("Pending Review", pending_records)
        with col3:
            st.metric("Processed", processed_records)

        # Calculate and show column differences
        key_columns = st.session_state.process_config["key_columns"]
        column_differences = calculate_column_differences(df, key_columns)
        if column_differences:
            st.write("**Balance Comparisons**")
            cols = st.columns(len(column_differences))
            for idx, (comparison, data) in enumerate(column_differences.items()):
                with cols[idx]:
                    col1, col2 = comparison.split('_vs_')
                    st.metric(
                        f"{col1} vs {col2}",
                        f"{data['difference']:.2f}%",
                        f"Diff: {abs(data['sum1'] - data['sum2']):,.2f}"
                    )

        # Add bulk processing button
        if st.button("Process All Records"):
            with st.spinner("Processing all records..."):
                for index, row in df.iterrows():
                    if index not in st.session_state.anomaly_results:
                        result = detect_anomalies(row.to_dict(), index)
                        if result:
                            df.at[index, 'Status'] = 'Analyzed'
                            df.at[index, 'Anomaly Detected'] = 'Yes' if result.get('final_assessment', {}).get('is_anomaly', True) else 'No'
                    else:
                        show_analysis_results(st.session_state.anomaly_results[index], index)
                st.success("Bulk processing complete!")

        # Display records table
        st.subheader("Records")
        for index, row in df.iterrows():
            with st.expander(f"Record {index + 1}: {' | '.join([f'{col}: {row[col]}' for col in key_columns])}"):
                if index not in st.session_state.anomaly_results:
                    if st.button("Detect Anomaly", key=f"detect_{index}"):
                        with st.spinner("Analyzing..."):
                            result = detect_anomalies(row.to_dict(), index)
                            if result:
                                df.at[index, 'Status'] = 'Analyzed'
                                show_analysis_results(result, index)
                else:
                    show_analysis_results(st.session_state.anomaly_results[index], index)

def main():
    st.title("AI Agent Powered Reconciliation")
    
    # Initialize session state
    initialize_session_state()
    
    # Show process setup or dashboard based on current page
    if st.session_state.current_page == 'setup':
        create_reconciliation_process()
    elif st.session_state.current_page == 'dashboard':
        show_process_config()
        show_dashboard()

if __name__ == "__main__":
    main()


#"""
#:Anomaly result: {'statistical_analysis': {'score': 1.0, 'is_anomaly': False}, 'llm_analysis': {'anomalies': [{'type': 'Break', 'confidence': '0.95', 'reasoning': 'The GL_Balance is not equal to the Ihub_Balance, indicating a discrepancy in the records.', 'impact': 'This could lead 
#to incorrect financial reporting and decision making.'}], 'overall_assessment': 'There is a significant anomaly detected in the record. The GL_Balance does not match the Ihub_Balance, indicating a potential discrepancy in the records. Further investigation is recommended to resolve this issue.'}, 'final_assessment': {'is_anomaly': True, 'confidence_score': 0.22499999999999998, 'timestamp': '2025-03-26T12:13:39.221397'}}, Fix suggestion: {'root_cause': 'The GL_Balance does not match the Ihub_Balance, indicating a potential discrepancy in the records.', 'recommended_fix': 'Review and reconcile the GL_Balance and Ihub_Balance to ensure accuracy.', 'implementation_steps': ['Verify that all transactions have been recorded correctly in both systems.', 'Check for any manual entries or adjustments that may have caused the discrepancy.', 'Contact the relevant parties to obtain additional information or clarification if necessary.'], 'risk_assessment': "The potential risk associated with this issue is significant, as it could lead to incorrect financial reporting and decision making. It is important to address this issue promptly to minimize any negative impact on the organization's finances."}
#"""