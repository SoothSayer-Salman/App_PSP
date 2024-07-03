import streamlit as st
import pandas as pd
import numpy as np
import scipy.optimize as sco
import joblib
import plotly.graph_objects as go

# Load the saved models and scalers
model_LR = joblib.load('LR_MODEL.pkl')
model_svr = joblib.load('SVR_MODEL.pkl')
LR_scalerx = joblib.load('LR_scalerx.pkl')
LR_scalery = joblib.load('LR_scalery.pkl')
svr_scalerx = joblib.load('SVR_scalerx.pkl')
svr_scalery = joblib.load('SVR_scalery.pkl')

# Define the mean absolute percentage error function
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Function to calculate profit and predictions
def ProfitFunc(df):
    df_svm = df[['Penetration', 'DMRate', 'GMRate', 'EconomicIndicator', 'Christmas', 'SeasonalIndicator']]
    df_svm_scaled = svr_scalerx.transform(df_svm)
    svrpred = model_svr.predict(df_svm_scaled)
    svrpreds = svr_scalery.inverse_transform(pd.DataFrame(svrpred)).flatten()  # Rescale SVR predictions

    df_linear = df[['EconomicIndicator', 'Christmas', 'SeasonalIndicator']]
    df_linear_scaled = LR_scalerx.transform(df_linear)
    linpred = model_LR.predict(df_linear_scaled)
    linpreds = LR_scalery.inverse_transform(linpred.reshape(-1, 1)).flatten() 

    EstimatedRevenue = linpreds + svrpreds
    df['TotDisc'] = EstimatedRevenue * df['Penetration']
    x1 = (EstimatedRevenue + df['TotDisc']) * df['GMRate'] - df['TotDisc']

    return -x1, linpreds.tolist(), svrpreds.tolist()


# Function to calculate profit and predictions
def ProfitFunc_all(X, df):
    df['Penetration'] = X[0]
    df['DMRate'] = X[1]
    
    df_svm = df[['Penetration', 'DMRate', 'GMRate', 'EconomicIndicator', 'Christmas', 'SeasonalIndicator']]
    df_svm_scaled = svr_scalerx.transform(df_svm)
    svrpred = model_svr.predict(df_svm_scaled)
    svrpreds = svr_scalery.inverse_transform(svrpred.reshape(-1, 1)).flatten()
    
    df_linear = df[['EconomicIndicator', 'Christmas', 'SeasonalIndicator']]
    df_linear_scaled = LR_scalerx.transform(df_linear)
    linpred = model_LR.predict(df_linear_scaled)
    linpreds = LR_scalery.inverse_transform(linpred.reshape(-1, 1)).flatten() 

    EstimatedRevenue = linpreds + svrpreds
    df['TotDisc'] = EstimatedRevenue * df['Penetration']
    x1 = (EstimatedRevenue + df['TotDisc']) * df['GMRate'] - df['TotDisc']
    
    return -x1, linpreds, svrpreds


# Function to find maxima
def findMaxima(data):
    guess = [0.01, 0.1]  # Initial guess for Penetration and DMRate
    constraints = ()
    args = (data,)
    bounds = [(0.01, 0.3), (0.1, 0.35)]  # Bounds for Penetration and DMRate
    
    res = sco.minimize(lambda X, *args: ProfitFunc_all(X, *args)[0], x0=guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if res.success:
        _, linpreds, svrpreds = ProfitFunc_all(res.x, data)
        estimated_revenue = linpreds + svrpreds
        
        return {
            "Estimated_revenue": estimated_revenue,
            "Estimated_Profit": -res.fun,
            "Penetration": res.x[0],
            "DMRate": res.x[1],
            "LR_prediction": linpreds,
            "SVR_prediction": svrpreds,
            "message": res.message
        }
    else:
        return {
            "success": res.success,
            "message": res.message
        }


# Streamlit app
def main():
    page_bg="""      
      <style>
        [data-testid="stAppViewContainer"]{
            background-color: #e5e5f7;
           
        }
        .sidebar .sidebar-content {
            background: Black;
            opacity: 0.8;
        }
        .stApp {
            color: green;
        }
        h1, h2, h3 {
            color: green;
        }
      label {
        color: #000000;
    }
    .css-1n76uvr {
        color: #000000;
    }
    .user_input_form {  /* Targeting user input form */
            background-color: #130F0F;  /* Change this color to your desired form background color */
            padding: 20px;
            border-radius: 10px;
            label:#000000;

        }
        </style>
        """
    st.markdown(page_bg,unsafe_allow_html=True)

    st.title("Discount Optimization")
    
    st.subheader("Option: Enter Values for Model Columns")
    
    # User input form
    with st.form("user_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            penetration_min = st.number_input("Penetration Min", min_value=0.0, max_value=1.0, step=0.01)
            penetration_max = st.number_input("Penetration Max", min_value=0.0, max_value=1.0, step=0.01)
            dmrate_min = st.number_input("DMRate Min", min_value=0.0, max_value=1.0, step=0.01)
            dmrate_max = st.number_input("DMRate Max", min_value=0.0, max_value=1.0, step=0.01)
        with col2:
            gm_rate = st.number_input("GM Rate", min_value=0.0, step=0.01)
            economic_indicator = st.number_input("Economic Indicator", min_value=0.0, step=0.01)
            christmas = st.number_input("Christmas", min_value=0.0, max_value=1.0, step=0.01)
            seasonal_indicator = st.number_input("Seasonal Indicator", min_value=0.0, max_value=1.0, step=0.01)
        
        submit_button = st.form_submit_button("Submit")
    
    if submit_button:
        # Generate Penetration and DMRate values
        Penetration = np.arange(penetration_min, penetration_max, 0.005)
        DMRate = np.arange(dmrate_min, dmrate_max, 0.005)

        # Create DataFrame for combinations
        A = pd.DataFrame({'Penetration': Penetration})
        B = pd.DataFrame({'DMRate': DMRate})
        A['key'] = 1
        B['key'] = 1
        df = pd.merge(A, B).drop('key', axis=1)
        df['SeasonalIndicator'] = seasonal_indicator
        df['Christmas'] = christmas
        df['EconomicIndicator'] = economic_indicator
        df['GMRate'] = gm_rate
# Calculate results for each combination
        results_list = []
        for i in range(df.shape[0]):
            result = findMaxima(df.iloc[i:i+1].copy())
            results_list.append(result)
        
        # Convert results to a DataFrame
        results_df = pd.DataFrame(results_list)
        st.write("Optimal Penetration and DMRate:")
        results_df = results_df.applymap(lambda x: x[0] if isinstance(x, list) else x)
        st.write(results_df.iloc[[0]])
        
        x1_list, linpreds_list, svrpreds_list = [], [], []
        for i in range(df.shape[0]):
            # Extract the row as a DataFrame and reset the index
            row = df.iloc[[i]].reset_index(drop=True)
            x1, linpreds, svrpreds = ProfitFunc(row.copy())
            x1_list.append(x1[0])  # Append the first element of x1
            linpreds_list.append(linpreds[0])
            svrpreds_list.append(svrpreds[0])

        # After the loop, assign lists to DataFrame columns
        df['x1'] = x1_list
        df['LR_prediction'] = linpreds_list
        df['SVR_prediction'] = svrpreds_list

        # Display table with all combinations
        st.write("Table showing all combinations with predictions and profit:")
        st.write(df)

        # Option to download the results as CSV
        csv_user_input = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv_user_input,
            file_name='results_from_user_input.csv',
            mime='text/csv',
        )

        # Plotting the data
        fig = go.Figure(data=[go.Scatter3d(
            x=df['Penetration'],
            y=df['DMRate'],
            z=df['x1'],
            mode='markers',
            marker=dict(
                size=5,
                color=df['x1'],                # set color to the profit
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )
        )])

        fig.update_layout(
            title='3D Plot of Estimated Profit vs Penetration and DMRate',
            scene=dict(
                xaxis_title='Penetration',
                yaxis_title='DMRate',
                zaxis_title='Estimated Profit'
            )
        )

        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
