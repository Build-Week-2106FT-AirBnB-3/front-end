# -*- coding: utf-8 -*-

# https://towardsdatascience.com/build-a-machine-learning-simulation-tool-with-dash-b3f6fd512ad6

# We start with the import of standard ML librairies
import pandas as pd
import numpy as np

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# We add all Plotly and Dash necessary librairies
import plotly.graph_objects as go
import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('clean_data.csv')

# importing our model
# target='price'
# X = df.drop(columns=target)
# y = df[target]

# # Let's split into a test and
# X_train, X_t, y_train, y_t = train_test_split(X,y, test_size=.2, random_state=7)

# # Let's split our test data into validation and test
# X_val, X_test, y_val, y_test = train_test_split(X_t,y_t, test_size=.2, random_state=7)


# model = make_pipeline(OneHotEncoder(use_cat_names=True),
#                          SimpleImputer(),
#                          RandomForestRegressor(random_state=70))
# model.fit(X_train,y_train)

infile = open('random_forest_model', 'rb')
model = pickle.load(infile)
infile.close()

# # # We create a DataFrame to store the features' importance and their corresponding label
f_impor = model.named_steps['randomforestregressor'].feature_importances_
col_names = model.named_steps['onehotencoder'].get_feature_names()
df_feature_importances = pd.DataFrame(f_impor, columns=["Importance"], index=col_names)
df_feature_importances = df_feature_importances.sort_values("Importance", ascending=False).head(10)


# Create the bar chart and limit it to the top 10 features


# # We create a Features Importance Bar Chart
fig_features_importance = go.Figure()
fig_features_importance.add_trace(go.Bar(x=df_feature_importances.index,
                                         y=df_feature_importances["Importance"],
                                         marker_color='rgb(171, 226, 251)')
                                 )
fig_features_importance.update_layout(title_text='<b>Features Importance of the model<b>', title_x=0.5)
# The command below can be activated in a standard notebook to display the chart
# fig_features_importance.show()

# We record the name, min, mean and max of the three most important features
dropdown_1_label = df_feature_importances.index[0]
dropdown_1_min = round(df[dropdown_1_label].min(),5)
dropdown_1_mean = round(df[dropdown_1_label].mean(),5)
dropdown_1_max = round(df[dropdown_1_label].max(),5)

dropdown_2_label = df_feature_importances.index[1]
dropdown_2_min = round(df[dropdown_2_label].min(),5)
dropdown_2_mean = round(df[dropdown_2_label].mean(),5)
dropdown_2_max = round(df[dropdown_2_label].max(),5)

dropdown_3_label = df_feature_importances.index[5]
dropdown_3_min = round(df[dropdown_3_label].min(),5)
dropdown_3_mean = round(df[dropdown_3_label].mean(),5)
dropdown_3_max = round(df[dropdown_3_label].max(),5)


###############################################################################

app = dash.Dash()

# The page structure will be:
#    Features Importance Chart
#    <H4> Feature #1 name
#    Slider to update Feature #1 value
#    <H4> Feature #2 name
#    Slider to update Feature #2 value
#    <H4> Feature #3 name
#    Slider to update Feature #3 value
#    <H2> Updated Prediction
#    Callback fuction with Sliders values as inputs and Prediction as Output

# We apply basic HTML formatting to the layout
app.layout = html.Div(style={'textAlign': 'center', 'width': '800px', 'font-family': 'Verdana'},
                      
                    children=[

                        # The same logic is applied to the following names / sliders
                        html.H1(children="Simulation Tool"),
                        
                        #Dash Graph Component calls the fig_features_importance parameters
                        dcc.Graph(figure=fig_features_importance),
                        
                        # We display the most important feature's name
                        html.H4(children=dropdown_1_label),

                        # The Dash Slider is built according to Feature #1 ranges
                        dcc.Slider(
                            id='X1_slider',
                            min=dropdown_1_min,
                            max=dropdown_1_max,
                            step=0.029311,
                            value=dropdown_1_mean,
                            marks={i: '{}°'.format(i) for i in np.arange(dropdown_1_min, dropdown_1_max)}
                            ),

                        # The same logic is applied to the following names / sliders
                        html.H4(children=dropdown_2_label),

                        dcc.Slider(
                            id='X2_slider',
                            min=dropdown_2_min,
                            max=dropdown_2_max,
                            step=0.080384,
                            value=dropdown_2_mean,
                            marks={i: '{}°'.format(i) for i in np.arange(dropdown_2_min, dropdown_2_max)}
                        ),

                        html.H4(children=dropdown_3_label),

                        dcc.Slider(
                            id='X3_slider',
                            min=dropdown_3_min,
                            max=dropdown_3_max,
                            step=0.6,
                            value=dropdown_3_mean,
                            marks={i: '{}people'.format(i) for i in np.arange(dropdown_2_min, dropdown_2_max)},
                        ),
                        
                        # The prediction result will be displayed and updated here
                        html.H2(id="prediction_result")

                    ])

# The callback function will provide one "Ouput" in the form of a string (=children)
@app.callback(Output(component_id="prediction_result",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
              [Input("X1_slider","value"), Input("X2_slider","value"), Input("X3_slider","value")])

# The input variable are set in the same order as the callback Inputs
def update_prediction(X1, X2, X3):

    # We create a NumPy array in the form of the original features
    # ["Pressure","Viscosity","Particles_size", "Temperature","Inlet_flow", "Rotating_Speed","pH","Color_density"]
    # Except for the X1, X2 and X3, all other non-influencing parameters are set to their mean
    input_X = np.array([258668827,
                        1,
                        1,
                        1,
                        X1,
                        X2,
                        X3,
                        df["bedrooms"].mean(),
                        df['beds'].mean(),
                        df['number_of_reviews'].mean(),
                        df["review_scores_rating"].mean(),
                        1,
                        1]).reshape(1, -1)

    
    # Prediction is calculated based on the input_X array
    prediction = model.named_steps['randomforestregressor'].predict(input_X)
    
    # And retuned to the Output of the callback function
    return "Prediction in Yen: {}".format(round(prediction[0]))
    # return 'this is working'

if __name__ == "__main__":
    app.run_server()