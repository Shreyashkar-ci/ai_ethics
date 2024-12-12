from ethics import FairnessMetrics, GroupFairnessScore, GroupFairnessAnalyzer

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
import json

def convert_np_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_np_floats(v) for k, v in obj.items()}
    elif isinstance(obj, np.float64):
        return float(obj)
    return obj

def assess_bias(df):
    # Bias thresholds dictionary
    bias_thresholds = {
        'Demographic Parity': {'lower': -0.1, 'upper': 0.1},
        'Equal Opportunity': {'lower': -0.1, 'upper': 0.1},
        'Predictive Parity': {'lower': 0.8, 'upper': 1.2},
        'JS Divergence': {'lower': 0, 'upper': 0.1},
        'Disparate Impact': {'lower': 0.8, 'upper': 1.2},
        'Equal Opportunity Difference': {'lower': -0.1, 'upper': 0.1},
        'Equalized Odds': {'lower': -0.1, 'upper': 0.1},
        'Statistical Parity Difference': {'lower': -0.1, 'upper': 0.1},
        'Treatment Equality': {'lower': 0.8, 'upper': 1.2},
        'Calibration by Group': {'lower': -0.1, 'upper': 0.1},
        'Conditional Parity': {'lower': -0.1, 'upper': 0.1},
        'Raw Outcome Disparity': {'lower': -0.1, 'upper': 0.1}
    }
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Function to determine bias
    def check_bias(row):
        metric = row['metric']
        value = row['metric_outcome']
        
        # Check if the metric exists in our thresholds
        if metric not in bias_thresholds:
            return 'Unknown Metric'
        
        thresholds = bias_thresholds[metric]
        
        # Check if the value is within the acceptable range
        if thresholds['lower'] <= value <= thresholds['upper']:
            return 'Fair'
        else:
            return 'Biased'
    
    # Apply bias check
    result_df['Bias'] = result_df.apply(check_bias, axis=1)
    
    return result_df

def main():
    st.title("AI Ethics Regulatory and Fairness Requirement")

    # Initialize session state for configurations
    if 'df' not in st.session_state:
        st.session_state.df = None
        st.session_state.ground_truth_col = None
        st.session_state.predictions_col = None
        st.session_state.sensitive_feature_cols = []
        st.session_state.statistical_target_feature = None
        st.session_state.analysis_results = None

    # Add CSV file uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # If a new file is uploaded, reset session state
    if uploaded_file is not None and uploaded_file != st.session_state.df:
        # Read the uploaded CSV file
        st.session_state.df = pd.read_csv(uploaded_file)
        # Reset other session state variables
        st.session_state.ground_truth_col = None
        st.session_state.predictions_col = None
        st.session_state.sensitive_feature_cols = []
        st.session_state.statistical_target_feature = None
        st.session_state.analysis_results = None

    # Proceed only if a DataFrame is loaded
    if st.session_state.df is not None:
        # Sidebar for configuration
        st.sidebar.header("Fairness Analysis Configuration")
        
        # Select ground truth column
        st.session_state.ground_truth_col = st.sidebar.selectbox(
            "Select Ground Truth Column", 
            st.session_state.df.columns.tolist(),
            index=st.session_state.df.columns.tolist().index(st.session_state.ground_truth_col) 
            if st.session_state.ground_truth_col in st.session_state.df.columns 
            else 0
        )
        
        # Select predictions column
        st.session_state.predictions_col = st.sidebar.selectbox(
            "Select Predictions Column", 
            st.session_state.df.columns.tolist(),
            index=st.session_state.df.columns.tolist().index(st.session_state.predictions_col) 
            if st.session_state.predictions_col in st.session_state.df.columns 
            else 0
        )
        
        # Select sensitive features
        st.session_state.sensitive_feature_cols = st.sidebar.multiselect(
            "Select Sensitive Features", 
            st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist(),
            default=st.session_state.sensitive_feature_cols
        )
        
        # Select statistical target feature
        st.session_state.statistical_target_feature = st.sidebar.selectbox(
            "Select Statistical Target Feature", 
            st.session_state.sensitive_feature_cols,
            index=st.session_state.sensitive_feature_cols.index(st.session_state.statistical_target_feature) 
            if st.session_state.statistical_target_feature in st.session_state.sensitive_feature_cols 
            else 0
        )
        
        # Perform analysis button
        if st.sidebar.button("Perform Fairness Analysis") or st.session_state.analysis_results is not None:
            # Prepare data for analysis
            ground_truth = list(st.session_state.df[st.session_state.ground_truth_col])
            predictions = list(st.session_state.df[st.session_state.predictions_col])
            
            # Prepare sensitive features
            sensitive_features = st.session_state.df[st.session_state.sensitive_feature_cols]
            
            # Compute fairness metrics
            fairness_metrics = FairnessMetrics(predictions, ground_truth)

            group_analyzer = GroupFairnessAnalyzer(
                sensitive_features,
                statistical_target_feature=st.session_state.statistical_target_feature
            )

            report = group_analyzer.generate_fairness_report(
                fairness_metrics,
                features_to_analyze=st.session_state.sensitive_feature_cols
            )

            data = report

            group_analyzer = GroupFairnessScore(
                sensitive_features,
                statistical_target_feature=st.session_state.statistical_target_feature
            )

            score_report = group_analyzer.generate_fairness_report(
                fairness_metrics,
                features_to_analyze=st.session_state.sensitive_feature_cols
            )

            # Store results in session state
            st.session_state.analysis_results = {
                'data': data,
                'score_report': score_report
            }

        # If analysis results exist, proceed with visualization
        if st.session_state.analysis_results is not None:
            data = st.session_state.analysis_results['data']
            score_report = st.session_state.analysis_results['score_report']

            # Extracting unique feature names and metric names for the dropdowns
            features = data['feature'].unique()
            metrics = data['metric'].unique()

            # Assess bias on score report
            scored_data = assess_bias(score_report)

            # Dropdowns for feature and metric selection
            selected_feature = st.selectbox("Select Feature Name:", features)
            selected_metric = st.selectbox("Select Metric Name:", metrics)

            # Displaying the final score for the selected feature and metric
            st.subheader("Final Score")
            scored_filtered = scored_data[(scored_data['feature'] == selected_feature) & (scored_data['metric'] == selected_metric)]
            if not scored_filtered.empty:
                final_score = scored_filtered.iloc[0]['metric_outcome']
                def color_bias(val):
                    color = 'background-color: lightgreen' if val == 'Fair' else 'background-color: lightcoral'
                    return color

                styled_df = scored_filtered.style.map(color_bias, subset=['Bias'])

                st.dataframe(styled_df)
            else:
                st.warning("No score available for the selected feature and metric.")

            # Filter data based on selections
            filtered_data = data[(data['feature'] == selected_feature) & (data['metric'] == selected_metric)]

            if not filtered_data.empty:
                # Parse the 'group_rates' column
                group_rates = filtered_data.iloc[0]['group_rates']
                try:
                    dict_str = json.dumps(convert_np_floats(group_rates))
                    group_rates = json.loads(dict_str)
                except (ValueError, SyntaxError):
                    st.error("Error parsing group rates.")
                    group_rates = {}

                if group_rates:
                          # Handle different metric-specific data structures
                    if selected_metric == 'Equalized Odds':
                        # Separate TPR and FPR rates
                        tpr_rates = group_rates.get('tpr_rates', {})
                        fpr_rates = group_rates.get('fpr_rates', {})

                        if tpr_rates and fpr_rates:
                            tpr_df = pd.DataFrame(list(tpr_rates.items()), columns=['Group', 'TPR']).sort_values(by='TPR', ascending=False)
                            fpr_df = pd.DataFrame(list(fpr_rates.items()), columns=['Group', 'FPR']).sort_values(by='FPR', ascending=False)

                            # Display TPR and FPR data
                            st.write("True Positive Rates:")
                            st.write(tpr_df)
                            st.write("False Positive Rates:")
                            st.write(fpr_df)

                            # Two bar charts side by side for TPR and FPR using Plotly
                            fig = make_subplots(rows=1, cols=2, subplot_titles=("True Positive Rates", "False Positive Rates"))

                            fig.add_trace(
                                go.Bar(x=tpr_df['TPR'], y=tpr_df['Group'], orientation='h',
                                    marker=dict(color=px.colors.qualitative.Pastel2), name='TPR'),
                                row=1, col=1
                            )

                            fig.add_trace(
                                go.Bar(x=fpr_df['FPR'], y=fpr_df['Group'], orientation='h',
                                    marker=dict(color=px.colors.qualitative.Pastel2), name='FPR'),
                                row=1, col=2
                            )

                            fig.update_layout(height=600, width=1000, title_text="Equalized Odds", showlegend=False)
                            st.plotly_chart(fig)

                        else:
                            st.warning("TPR and FPR rates are missing or invalid.")

                    elif selected_metric == 'Calibration by Group':
                        # Separate predicted_positive_rate, actual_positive_rate, and calibration_error
                        calibration_data = {
                            group: {
                                'Predicted Positive Rate': value.get('predicted_positive_rate', 0),
                                'Actual Positive Rate': value.get('actual_positive_rate', 0),
                                'Calibration Error': value.get('calibration_error', 0)
                            } for group, value in group_rates.items()
                        }
                        calibration_df = pd.DataFrame(calibration_data).T.reset_index()
                        calibration_df.columns = ['Group', 'Predicted Positive Rate', 'Actual Positive Rate', 'Calibration Error']

                        st.write(calibration_df)

                        # Line plot for predicted vs actual positive rates using Plotly
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=calibration_df['Group'], y=calibration_df['Predicted Positive Rate'],
                                                mode='lines+markers', name='Predicted Positive Rate'))
                        fig.add_trace(go.Scatter(x=calibration_df['Group'], y=calibration_df['Actual Positive Rate'],
                                                mode='lines+markers', name='Actual Positive Rate'))

                        fig.update_layout(title=f"Calibration for {selected_feature}",
                                        xaxis_title="Group",
                                        yaxis_title="Rate",
                                        height=500, width=800)
                        st.plotly_chart(fig)

                        # Bar chart for calibration error using Plotly
                        fig = px.bar(calibration_df, x='Group', y='Calibration Error', color='Group',
                                    title=f"Calibration Error for {selected_feature}", height=500, width=800,
                                    color_discrete_sequence=px.colors.qualitative.Pastel2)
                        st.plotly_chart(fig)

                    elif selected_metric in ['Demographic Parity', 'Statistical Parity Difference', 'JS Divergence', 'Raw Outcome Disparity', 'Disparate Impact']:
                        # Horizontal Bar Chart using Plotly
                        group_rates_df = pd.DataFrame(list(group_rates.items()), columns=['Group', 'Value']).sort_values(by='Value', ascending=False)
                        st.write(group_rates_df)

                        fig = px.bar(group_rates_df, x='Value', y='Group', orientation='h',
                                    color='Group', title=f"Horizontal Bar Chart for {selected_feature} - {selected_metric}",
                                    height=500, width=800,
                                    color_discrete_sequence=px.colors.qualitative.Pastel2)
                        st.plotly_chart(fig)

                    else:
                        # Default to table and bar chart
                        group_rates_df = pd.DataFrame(list(group_rates.items()), columns=['Group', 'Value']).sort_values(by='Value', ascending=False)
                        st.write(group_rates_df)

                        fig = px.bar(group_rates_df, x='Value', y='Group', orientation='h',
                                    color='Group', title=f"Bar Chart for {selected_feature} - {selected_metric}",
                                    height=500, width=800,
                                    color_discrete_sequence=px.colors.qualitative.Pastel2)
                        st.plotly_chart(fig)
                else:
                    st.warning("Group rates data is empty or invalid for the selected combination.")
            else:
                st.warning("No data available for the selected combination. Please choose a different feature or metric.")
    else:
        st.warning("Upload csv to begin analysis!")

if __name__ == "__main__":
    main()
