import streamlit as st
import pandas as pd
import numpy as np
import scipy.optimize as sco
import joblib
import plotly.graph_objects as go

# Load the model and scalers
model_svr = joblib.load('new_SVR_Model.pkl')
svr_scalerx = joblib.load('Nsvr_scalarx.pkl')
svr_scalery = joblib.load('Nsvr_scalary.pkl')

# Fixed values
GMRate = 0.477155
DMPromo = 0.207353
DMOther = 0.113064
PromoPenetration = 0.110759
OtherPenetration = 0.010521
EconomicIndicator = 73465
SF = 1.133904

def ProfitFunc_all(X):
    CouponPenetration = X[0]
    DMCoupon = X[1]

    data_copy = pd.DataFrame({
        'CouponPenetration': [CouponPenetration],
        'DMCoupon': [DMCoupon],
        'GMRate': [GMRate],
        'DMPromo': [DMPromo],
        'DMOther': [DMOther],
        'PromoPenetration': [PromoPenetration],
        'OtherPenetration': [OtherPenetration],
        'EconomicIndicator': [EconomicIndicator],
        'SF': [SF]
    })

    df_svm = data_copy[['GMRate', 'DMCoupon', 'DMPromo', 'DMOther', 'CouponPenetration', 'PromoPenetration', 'OtherPenetration', 'EconomicIndicator', 'SF']]
    df_svm_scaled = svr_scalerx.transform(df_svm)
    svrpred = model_svr.predict(df_svm_scaled)
    svrpreds = svr_scalery.inverse_transform(svrpred.reshape(-1, 1)).flatten()
    Estimated_Profit = svrpreds

    return -Estimated_Profit

# Function to find maxima
def findMaxima(penetration_min, penetration_max, dmrate_min, dmrate_max):
    guess = [(penetration_min + penetration_max) / 2, (dmrate_min + dmrate_max) / 2]

    constraints = ()
    bounds = [(penetration_min, penetration_max), (dmrate_min, dmrate_max)]

    res = sco.minimize(ProfitFunc_all, x0=guess, method='COBYLA', bounds=bounds, constraints=constraints)

    if res.success:
        CouponPenetration, DMCoupon = res.x
        data_copy = pd.DataFrame({
            'CouponPenetration': [CouponPenetration],
            'DMCoupon': [DMCoupon],
            'GMRate': [GMRate],
            'DMPromo': [DMPromo],
            'DMOther': [DMOther],
            'PromoPenetration': [PromoPenetration],
            'OtherPenetration': [OtherPenetration],
            'EconomicIndicator': [EconomicIndicator],
            'SF': [SF]
        })

        df_svm = data_copy[['GMRate', 'DMCoupon', 'DMPromo', 'DMOther', 'CouponPenetration', 'PromoPenetration', 'OtherPenetration', 'EconomicIndicator', 'SF']]
        df_svm_scaled = svr_scalerx.transform(df_svm)
        svrpred = model_svr.predict(df_svm_scaled)
        svrpreds = svr_scalery.inverse_transform(svrpred.reshape(-1, 1)).flatten()
        Estimated_Profit = svrpreds

        return {
            "success": res.success,
            "Estimated_Profit": Estimated_Profit[0],
            "CouponPenetration": CouponPenetration,
            "DMCoupon": DMCoupon,
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
        [data-testid="stAppViewContainer"] {
            background-color: #e5e5f7;
            color: black;
        }
        .sidebar .sidebar-content {
            background: black;
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
        .user_input_form {
            background-color: #130F0F;
            padding: 20px;
            border-radius: 10px;
            label: #000000;
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
            penetration_min = st.number_input("Coupon Penetration Min", min_value=0.0, max_value=1.0, step=0.01)
            penetration_max = st.number_input("Coupon Penetration Max", min_value=0.0, max_value=1.0, step=0.01)
            dmrate_min = st.number_input("DMCoupon Min", min_value=0.0, max_value=1.0, step=0.01)
            dmrate_max = st.number_input("DMCoupon Max", min_value=0.0, max_value=1.0, step=0.01)
            # GMRate = st.number_input("GMRate", min_value=0.0, step=0.01)
            

        # with col2:
        #     DMPromo = st.number_input("DMPromo", min_value=0.0, step=0.01)
        #     DMOther = st.number_input("DMOther", min_value=0.0, step=0.01)
        #     PromoPenetration = st.number_input("PromoPenetration", min_value=0.0, step=0.01)
        #     OtherPenetration = st.number_input("OtherPenetration", min_value=0.0, step=0.01)
        #     SF = st.number_input("SF", min_value=0.0, step=0.01)
        
        # EconomicIndicator = st.number_input("EconomicIndicator", min_value=0.0)
           
        submit_button = st.form_submit_button("Submit")       

    if submit_button:
        with st.spinner('Calculations in progress...'):
            progress_bar = st.progress(0)

            # Adjust Penetration and DMRate values if min equals max
            Penetration = np.array([penetration_min]) if penetration_min == penetration_max else np.arange(penetration_min, penetration_max, 0.005)
            DMRate = np.array([dmrate_min]) if dmrate_min == dmrate_max else np.arange(dmrate_min, dmrate_max, 0.005)

            # Create DataFrame for combinations
            A = pd.DataFrame({'CouponPenetration': Penetration})
            B = pd.DataFrame({'DMCoupon': DMRate})
            A['key'] = 1
            B['key'] = 1
            df = pd.merge(A, B).drop('key', axis=1)
            df['GMRate'] = GMRate
            df['DMPromo'] = DMPromo
            df['DMOther'] = DMOther
            df['PromoPenetration'] = PromoPenetration
            df['OtherPenetration'] = OtherPenetration
            df['EconomicIndicator'] = EconomicIndicator
            df['SF'] = SF

            svrpreds_list = []
            total = df.shape[0]

            for i in range(df.shape[0]):
                progress_bar.progress((i + 1) / total)

                row = df.iloc[[i]].reset_index(drop=True)
                svrpred = ProfitFunc_all([row['CouponPenetration'][0], row['DMCoupon'][0]])
                svrpreds_list.append(-svrpred[0])

            df['Estimated_Profit'] = svrpreds_list

            st.write("Table showing all combinations with predictions and profit:")
            st.write(df)

            results = findMaxima(penetration_min, penetration_max, dmrate_min, dmrate_max)

            if results['success']:
                st.write("Optimal Coupon Penetration and DMCoupon:")
                st.write(f"Coupon Penetration: {results['CouponPenetration']:.5f}")
                st.write(f"DMCoupon: {results['DMCoupon']:.5f}")
                st.write(f"Estimated Profit: {results['Estimated_Profit']:.2f}")

                optimal_point = {
                    'CouponPenetration': results['CouponPenetration'],
                    'DMCoupon': results['DMCoupon'],
                    'Estimated_Profit': results['Estimated_Profit']
                }

                csv_user_input = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv_user_input,
                    file_name='results_from_user_input.csv',
                    mime='text/csv',
                )

                if penetration_min == penetration_max or dmrate_min == dmrate_max:
                    if penetration_min == penetration_max:
                        fig = go.Figure(data=go.Scatter(
                            x=df['DMCoupon'],
                            y=df['Estimated_Profit'],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=df['Estimated_Profit'],
                                colorscale='Viridis',
                                opacity=0.8
                            )
                        ))

                        fig.add_trace(go.Scatter(
                            x=[optimal_point['DMCoupon']],
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
                            title='2D Plot of Estimated_Profit vs DMCoupon',
                            xaxis_title='DMCoupon',
                            yaxis_title='Estimated_Profit'
                        )
                    elif dmrate_min == dmrate_max:
                        fig = go.Figure(data=go.Scatter(
                            x=df['CouponPenetration'],
                            y=df['Estimated_Profit'],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=df['Estimated_Profit'],
                                colorscale='Viridis',
                                opacity=0.8
                            )
                        ))

                        fig.add_trace(go.Scatter(
                            x=[optimal_point['CouponPenetration']],
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
                            title='2D Plot of Estimated_Profit vs CouponPenetration',
                            xaxis_title='CouponPenetration',
                            yaxis_title='Estimated_Profit'
                        )

                    st.plotly_chart(fig)
                else:
                    fig = go.Figure(data=go.Scatter3d(
                        x=df['CouponPenetration'],
                        y=df['DMCoupon'],
                        z=df['Estimated_Profit'],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=df['Estimated_Profit'],
                            colorscale='Viridis',
                            opacity=0.8
                        )
                    ))

                    fig.add_trace(go.Scatter3d(
                        x=[optimal_point['CouponPenetration']],
                        y=[optimal_point['DMCoupon']],
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
                        title='3D Plot of Estimated_Profit vs CouponPenetration and DMCoupon',
                        scene=dict(
                            xaxis_title='CouponPenetration',
                            yaxis_title='DMCoupon',
                            zaxis_title='Estimated_Profit'
                        )
                    )

                    st.plotly_chart(fig)
            else:
                st.write("Optimization failed.")
                st.write(f"Message: {results['message']}")

if __name__ == "__main__":
    main()

