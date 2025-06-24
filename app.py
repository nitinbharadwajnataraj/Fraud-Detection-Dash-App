
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import eda
import json
import requests
import pandas as pd
import io, base64
import plotly.express as px

df = eda.load_data()
overview = eda.get_data_overview(df)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
app.title = "Fraud Detection Dashboard"

sidebar = html.Div([
    html.H4("Dashboard", className="text-center mb-4"),
    html.Hr(),
    dbc.Nav([
        dbc.NavLink("🏠 Home", href="/", active="exact"),
        dbc.NavLink("📊 EDA", href="/eda", active="exact"),
        dbc.NavLink("📈 Models Outcome", href="/models", active="exact"),
        dbc.NavLink("🧪 Predict", href="/predict", active="exact"),
    ], vertical=True, pills=True),
], style={
    "position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "250px",
    "padding": "20px", "backgroundColor": "#8cbdee", "color": "white", "overflowY": "auto"
})

home_content = html.Div([
    html.Div([
        html.H1("Vehicle Insurance Claim Fraud Detection", className="mb-3"),
        html.P("This dashboard is part of an end-to-end machine learning project designed to detect fraudulent vehicle insurance claims.", className="lead"),
    ], style={"backgroundColor": "#8ec4a9", "color": "white", "padding": "40px", "borderRadius": "12px", "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"}),

    html.Br(),
    dbc.Row([
        dbc.Col(html.Img(src="/static/f1.jpg", style={"width": "100%", "borderRadius": "10px"}), md=6),
        dbc.Col(html.Img(src="/static/f2.jpg", style={"width": "100%", "borderRadius": "10px"}), md=6)
    ], className="mt-3 mb-4")
], className="p-4", style={"backgroundColor": "white", "borderRadius": "10px", "boxShadow": "0 0 12px rgba(0,0,0,0.08)"})

model_results = json.load(open("results/model_metrics.json"))

model_content = html.Div([
    html.H2("📈 Models Outcome", className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5(model, className="text-center")),
            dbc.CardBody([
                html.Ul([
                    html.Li(f"{metric.capitalize()}: {val}") for metric, val in result.items()
                ], className="list-unstyled text-center")
            ])
        ], className="shadow-sm mb-4 rounded"), width=4)
        for model, result in model_results.items()
    ])
], className="p-4", style={"backgroundColor": "white", "borderRadius": "10px"})

def get_available_models():
    try:
        res = requests.get("http://127.0.0.1:8000/")
        return res.json().get("available_models", [])
    except Exception as e:
        print("Model fetch failed:", e)
        return []

predict_content = html.Div([
    html.H2("🧪 Upload for Prediction", className="mb-4"),
    html.Div([
    html.H6("📄 Need a sample Test Data file?"),
    dbc.Button("⬇ Click here!", id="btn-sample-download", color="info", className="mb-3"),
    dcc.Download(id="download-sample-csv")
    ]),

    dcc.Upload(
        id="upload-data",
        children=html.Div(["📁 Drag and Drop or ", html.A("Select CSV or Excel File")]),
        style={
            "width": "100%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
            "textAlign": "center", "marginBottom": "10px"
        },
        multiple=False
    ),

    html.Div(id="upload-status", style={"marginBottom": "10px"}),

    dcc.Dropdown(
        id="model-selector",
        options=[{"label": m, "value": m} for m in get_available_models()],
        placeholder="Select a model for prediction",
        style={"marginBottom": "20px"}
    ),

    dbc.Button("🔍 Predict", id="predict-button", color="dark", className="mb-3", n_clicks=0),

    html.Div(id="prediction-summary"),
    html.Div(id="prediction-table", className="mb-3"),

    # Hidden components for download
    dcc.Store(id="stored-uploaded-data"),
    dcc.Store(id="stored-predicted-data"),
    html.Div([
        dbc.Button("⬇ Download Predictions", id="btn-download", color="success", style={"display": "none"}),
        dcc.Download(id="download-xlsx")
    ])
], className="p-4", style={"backgroundColor": "white", "borderRadius": "10px"})

eda_content = html.Div([
    html.H2("📊 Exploratory Data Analysis", className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("SHAPE", className="card-title text-center"),
                html.P(f"{overview['shape']}", className="card-text text-center")
            ])
        ], className="shadow-sm rounded", style={"height": "120px"}), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("MISSING VALUES", className="card-title text-center"),
                html.P(f"{overview['nulls']}", className="card-text text-center")
            ])
        ], className="shadow-sm rounded", style={"height": "120px"}), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("DUPLICATES", className="card-title text-center"),
                html.P(f"{overview['duplicates']}", className="card-text text-center")
            ])
        ], className="shadow-sm rounded", style={"height": "120px"}), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("FRAUD DISTRIBUTION", className="card-title text-center"),
                html.P(f"{overview['target_distribution']}", className="card-text text-center")
            ])
        ], className="shadow-sm rounded", style={"height": "120px"}), width=3),
    ], className="mb-4"),
    html.H4("📈 Vehicle Price vs Fraud"),
    dcc.Graph(figure=eda.get_categorical_distribution(df, "VehiclePrice"), className="mb-4"),
    html.H4("📉 Age vs Fraud"),
    dcc.Graph(figure=eda.get_numeric_distribution(df, "Age"), className="mb-4"),
    #html.H4("📌 Top Features Influencing Fraud"),
    #dcc.Graph(figure=eda.get_top_feature_importance_plot(df)),
    html.H4("📌 Top Features Influencing Fraud using PI"),
    dcc.Graph(figure=eda.get_top_feature_importance_PI(df)),
    html.H4("🔗 Correlation Matrix"),
    dcc.Graph(figure=eda.get_full_correlation_heatmap(df))
    #html.H4("🔗 Correlation Matrix"),
    #dcc.Graph(figure=eda.get_correlation_heatmap(df))
    
], className="p-4", style={"backgroundColor": "white", "borderRadius": "10px", "boxShadow": "0 0 10px rgba(0,0,0,0.1)"})

content = html.Div(id="page-content", style={
    "marginLeft": "270px", "padding": "2rem 2rem", "backgroundColor": "#ecf0f1", "minHeight": "100vh"
})

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page(pathname):
    if pathname == "/eda":
        return eda_content
    elif pathname == "/models":
        return model_content
    elif pathname == "/predict":
        return predict_content
    return home_content

@app.callback(
    Output("stored-uploaded-data", "data"),
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def handle_file_upload(contents, filename):
    if contents is None:
        return None, None
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, html.Div("❌ Unsupported file format.")
        return df.to_json(date_format='iso', orient='split'), html.Div("✅ File uploaded.")
    except Exception as e:
        return None, html.Div(f"❌ Error: {str(e)}")


@app.callback(
    Output("prediction-table", "children"),
    Output("prediction-summary", "children"),
    Output("stored-predicted-data", "data"),
    Output("btn-download", "style"),
    Input("predict-button", "n_clicks"),
    State("stored-uploaded-data", "data"),
    State("model-selector", "value")
)
def make_prediction(n_clicks, data, model_name):
    if n_clicks == 0 or data is None or model_name is None:
        return dash.no_update, dash.no_update, dash.no_update, {"display": "none"}

    try:
        df = pd.read_json(data, orient='split')
        payload = {
            "model_name": model_name,
            "columns": df.columns.tolist(),
            "rows": df.astype(str).values.tolist()
        }

        res = requests.post("http://127.0.0.1:8000/batch_predict", json=payload)
        if res.status_code != 200:
            return html.Div("❌ Prediction API error."), None, None, {"display": "none"}

        raw_preds = res.json()["predictions"]
        df["Prediction"] = ["Fraud" if p == 1 else "Not Fraud" for p in raw_preds]

        count_fraud = df["Prediction"].value_counts().get("Fraud", 0)
        count_not_fraud = df["Prediction"].value_counts().get("Not Fraud", 0)

        summary = html.Div([
            html.H5("🔍 Prediction Summary"),
            html.P(f"✅ Not Fraud: {count_not_fraud}"),
            html.P(f"🚨 Fraud: {count_fraud}")
        ], style={
            "backgroundColor": "#f8f9fa",
            "padding": "10px", "borderRadius": "8px",
            "marginBottom": "15px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
        })


        table = dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in df.columns],
            style_data_conditional=[
                {"if": {"filter_query": '{Prediction} eq "Fraud"'}, "backgroundColor": "#ffcccc"},
                {"if": {"filter_query": '{Prediction} eq "Not Fraud"'}, "backgroundColor": "#ccffcc"},
            ],
            style_table={"overflowX": "auto"},
            page_size=10
        )

        return table, summary, df.to_json(date_format="iso", orient="split"), {"display": "inline-block"}
    except Exception as e:
        return html.Div(f"❌ Prediction failed: {str(e)}"), None, None, {"display": "none"}

@app.callback(
    Output("download-xlsx", "data"),
    Input("btn-download", "n_clicks"),
    State("stored-predicted-data", "data"),
    prevent_initial_call=True
)
def download_predictions(n_clicks, data):
    if data is None:
        return dash.no_update
    df = pd.read_json(data, orient="split")
    return dcc.send_data_frame(df.to_excel, filename="predictions.xlsx", index=False)

@app.callback(
    Output("download-sample-csv", "data"),
    Input("btn-sample-download", "n_clicks"),
    prevent_initial_call=True
)
def download_sample_file(n_clicks):
    return dcc.send_file("static/Sample_Test_Data.csv")



if __name__ == "__main__":
    app.run(debug=True)
