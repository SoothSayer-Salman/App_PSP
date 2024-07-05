import streamlit as st
import pandas as pd
import numpy as np
import scipy.optimize as sco
import joblib
import plotly.graph_objects as go
import pickle
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.models import load_model

class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        predictions = self.model.predict(X, **kwargs)
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        return predictions

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))

with open('voting_ensemble_2.pkl', 'rb') as f:
    model_svr = pickle.load(f)

keras_model = load_model('final_model_1.h5')
keras_wrapper = KerasRegressorWrapper(keras_model)

for idx, (name, model) in enumerate(model_svr.estimators):
    if name == 'keras':
        model_svr.estimators[idx] = ('keras', keras_wrapper)

# Load the saved models and scalers
model_LR = joblib.load('model_LR.pkl')
LR_scalerx = joblib.load('scalarx_linear.pkl')
LR_scalery = joblib.load('scalary_linear.pkl')
svr_scalerx = joblib.load('scalarx.pkl')
svr_scalery = joblib.load('scalary.pkl')

# Function to calculate profit and predictions
def ProfitFunc(df):
    df_svm = df[['Penetration', 'SeasonalIndicator_ts', 'DMRate', 'GMRate', 'EconomicIndicator', 'Christmas']]
    df_svm_scaled = svr_scalerx.transform(df_svm)
    svrpred = model_svr.predict(df_svm_scaled)
    svrpreds = svr_scalery.inverse_transform(svrpred.reshape(-1, 1)).flatten()
    
    df_linear = df[['EconomicIndicator', 'SeasonalIndicator_ts', 'Christmas']]
    df_linear_scaled = LR_scalerx.transform(df_linear)
    linpred = model_LR.predict(df_linear_scaled)
    linpreds = LR_scalery.inverse_transform(linpred.reshape(-1, 1)).flatten() 

    EstimatedRevenue = linpreds + svrpreds
    df['TotDisc'] = EstimatedRevenue * df['Penetration']
    x1 = (EstimatedRevenue + df['TotDisc']) * df['GMRate'] - df['TotDisc']

    return x1, linpreds.tolist(), svrpreds.tolist()

import numpy as np
import pandas as pd
import scipy.optimize as sco

def ProfitFunc_all(X, data):
    data_copy = data.copy()
    data_copy['Penetration'] = X[0]
    data_copy['DMRate'] = X[1]
    
    df_svm = data_copy[['Penetration', 'SeasonalIndicator_ts', 'DMRate', 'GMRate', 'EconomicIndicator', 'Christmas']]
    df_svm_scaled = svr_scalerx.transform(df_svm)
    svrpred = model_svr.predict(df_svm_scaled)
    svrpreds = svr_scalery.inverse_transform(svrpred.reshape(-1, 1)).flatten()
    
    df_linear = data_copy[['EconomicIndicator', 'SeasonalIndicator_ts', 'Christmas']]
    df_linear_scaled = LR_scalerx.transform(df_linear)
    linpred = model_LR.predict(df_linear_scaled)
    linpreds = LR_scalery.inverse_transform(linpred.reshape(-1, 1)).flatten() 

    EstimatedRevenue = linpreds + svrpreds
    data_copy['TotDisc'] = EstimatedRevenue * data_copy['Penetration']
    x1 = (EstimatedRevenue + data_copy['TotDisc']) * data_copy['GMRate'] - data_copy['TotDisc']
    
    return -x1.sum(), linpreds, svrpreds

# Jacobian function
def model_function_jac(X, df):
    penetration, dmrate = X
    
    # Initialize perturbation values
    delta = 1e-5  # Slightly larger perturbation for numerical stability
    
    # Calculate profit at the current point
    f0, _, _ = ProfitFunc_all(X, df)
    
    # Partial derivative with respect to Penetration
    X1 = [penetration + delta, dmrate]
    f1, _, _ = ProfitFunc_all(X1, df)
    dfd_penetration = (f1 - f0) / delta
    
    # Partial derivative with respect to DMRate
    X2 = [penetration, dmrate + delta]
    f2, _, _ = ProfitFunc_all(X2, df)
    dfd_dmrate = (f2 - f0) / delta
    
    return np.array([dfd_penetration, dfd_dmrate])

def findMaxima(data, penetration_min, penetration_max, dmrate_min, dmrate_max):
    guess = [(penetration_min + penetration_max) / 2, (dmrate_min + dmrate_max) / 2]  # Use mid-point as initial guess
    
    constraints = ()
    args = (data,)
    bounds = [(penetration_min, penetration_max), (dmrate_min, dmrate_max)]
    
    # Define a function to print optimization progress
    def print_optimization_progress(xk):
        objective_value, linpreds, svrpreds = ProfitFunc_all(xk, data)
        estimated_revenue = linpreds + svrpreds
        print(f"Iteration: {print_optimization_progress.iteration}, Objective: {objective_value}, Penetration: {xk[0]}, DMRate: {xk[1]}")
        print_optimization_progress.iteration += 1
    
    # Initialize iteration count attribute
    print_optimization_progress.iteration = 0
    
    # Optimization method parameters
    options = {'disp': True, 'maxiter': 100, 'ftol': 1e-6}  # Example options, adjust as needed
    
    res = sco.minimize(lambda X, *args: ProfitFunc_all(X, *args)[0], x0=guess, args=args, method='SLSQP', jac=lambda X, *args: model_function_jac(X, *args), bounds=bounds, constraints=constraints, options=options, callback=print_optimization_progress)
    
    if res.success:
        _, linpreds, svrpreds = ProfitFunc_all(res.x, data)
        estimated_revenue = linpreds + svrpreds
        
        return {
            "Estimated_revenue": estimated_revenue,
            "Estimated_Profit": round(-res.fun, 2),
            "Penetration": res.x[0],
            "DMRate": res.x[1],
            "LR_prediction": np.round(linpreds, 2),
            "Ensembling_prediction": np.round(svrpreds, 2),
            "message": res.message
        }
    else:
        return {
            "success": res.success,
            "message": res.message
        }


# Streamlit app
def main():
    page_bg = """      
      <style>
        [data-testid="stAppViewContainer"]{
            background-color: #e5e5f7;
            color: black;           
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
    st.markdown(page_bg, unsafe_allow_html=True)

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
            seasonal_indicator = st.number_input("Seasonal Indicator", min_value=0.0, step=0.01)
        
        submit_button = st.form_submit_button("Submit")
    
    if submit_button:
        with st.spinner('Calculations in progress...'):
            # Add a progress bar
            progress_bar = st.progress(0)

            # Adjust Penetration and DMRate values if min equals max
            Penetration = np.array([penetration_min]) if penetration_min == penetration_max else np.arange(penetration_min, penetration_max, 0.005)
            DMRate = np.array([dmrate_min]) if dmrate_min == dmrate_max else np.arange(dmrate_min, dmrate_max, 0.005)

            # Create DataFrame for combinations
            A = pd.DataFrame({'Penetration': Penetration})
            B = pd.DataFrame({'DMRate': DMRate})
            A['key'] = 1
            B['key'] = 1
            df = pd.merge(A, B).drop('key', axis=1)
            df['SeasonalIndicator_ts'] = seasonal_indicator
            df['Christmas'] = christmas
            df['GMRate'] = gm_rate
            df['EconomicIndicator'] = economic_indicator
            
            # Calculate results for each combination
            x1_list, linpreds_list, svrpreds_list = [], [], []
            total = df.shape[0]

            for i in range(df.shape[0]):
                # Update progress bar
                progress_bar.progress((i + 1) / total)

                # Extract the row as a DataFrame and reset the index
                row = df.iloc[[i]].reset_index(drop=True)
                x1, linpreds, svrpreds = ProfitFunc(row.copy())
                x1_list.append(round(x1[0], 2))
                linpreds_list.append(np.round(linpreds[0], 2))
                svrpreds_list.append(np.round(svrpreds[0], 2))

            # After the loop, assign lists to DataFrame columns
            df['Profit'] = x1_list
            df['LR_prediction'] = linpreds_list
            df['Enesembling_prediction'] = svrpreds_list

            # Display table with all combinations
            st.write("Table showing all combinations with predictions and profit:")
            st.write(df)
            data=pd.DataFrame()
         
            data['SeasonalIndicator_ts'] = [seasonal_indicator]
            data['Christmas'] = [christmas]
            data['GMRate'] = [gm_rate]
            data['EconomicIndicator'] = [economic_indicator]

            # Append necessary columns with initial guesses
            data['Penetration'] = [penetration_min]
            data['DMRate'] = [dmrate_min]
            

            results_list = findMaxima(data.copy(), penetration_min, penetration_max, dmrate_min, dmrate_max)
                

            # Convert results to a DataFrame
            results_df = pd.DataFrame(results_list)
            st.write("Optimal Penetration and DMRate:")
            results_df = results_df.applymap(lambda x: x[0] if isinstance(x, list) else x)
            st.write(results_df.iloc[[0]])
            optimal_point = results_df.iloc[0]

            # Option to download the results as CSV
            csv_user_input = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download results as CSV",
                data=csv_user_input,
                file_name='results_from_user_input.csv',
                mime='text/csv',
            )

            if penetration_min == penetration_max or dmrate_min == dmrate_max:
                # Plotting 2D plot between Estimated Profit and respective Penetration or DMRate
                if penetration_min == penetration_max:
                    fig = go.Figure(data=go.Scatter(
                        x=df['DMRate'],
                        y=df['Profit'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=df['Profit'],
                            colorscale='Viridis',
                            opacity=0.8
                        )
                    ))
                    fig.update_layout(
                        title='2D Plot of Estimated Profit vs DMRate',
                        xaxis_title='DMRate',
                        yaxis_title='Estimated Profit'
                    )

                    fig.add_trace(go.Scatter(
                    # x=[optimal_point['Penetration']],
                    x=[optimal_point['DMRate']],
                    y=[optimal_point['Estimated_Profit']],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='circle'
                    ),
                    name='Optimal Point'
                ))
                elif dmrate_min == dmrate_max:
                    fig = go.Figure(data=go.Scatter(
                        x=df['Penetration'],
                        y=df['Profit'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=df['Profit'],
                            colorscale='Viridis',
                            opacity=0.8
                        )
                    ))

                    fig.add_trace(go.Scatter(
                    x=[optimal_point['Penetration']],
                    # y=[optimal_point['DMRate']],
                    y=[optimal_point['Estimated_Profit']],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='circle'
                    ),
                    name='Optimal Point'
                ))
                    fig.update_layout(
                        title='2D Plot of Estimated Profit vs Penetration',
                        xaxis_title='Penetration',
                        yaxis_title='Estimated Profit'
                    )

                st.plotly_chart(fig)

            else:
                # Plotting 3D plot
                fig = go.Figure(data=[go.Scatter3d(
                    x=df['Penetration'],
                    y=df['DMRate'],
                    z=df['Profit'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=df['Profit'],  # set color to the profit
                        colorscale='Viridis',  # choose a colorscale
                        opacity=0.8
                    )
                )])

                # Add optimal point to the plot
                fig.add_trace(go.Scatter3d(
                    x=[optimal_point['Penetration']],
                    y=[optimal_point['DMRate']],
                    z=[optimal_point['Estimated_Profit']],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='diamond'
                    ),
                    name='Optimal Point'
                ))

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
#
