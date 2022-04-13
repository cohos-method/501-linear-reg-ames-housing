import dash
from dash import dcc,html
from dash.dependencies import Input, Output, State
option = [{'label':'NYC','value':'NYC'}, {'label':'MTL','value':'MTL'}, {'label':'SF','value':'SF'}]
AgeCatList= ['55-59', '80 or older', '65-69', '75-79', '40-44', '70-74', '60-64', '50-54', '45-49', '18-24', '35-39', '30-34', '25-29']
AgeCatList.sort()

def buildOptionsDict(lbl):
    opt = []
    for i in range(len(lbl)):
        d = {}
        d['label'] = lbl[i]
        d['value'] = lbl[i]
        opt.append(d)
    return opt

option = buildOptionsDict(AgeCatList)

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Dropdown(options=option, value='NYC', id='demo-dropdown'),
    html.Div(id='dd-output-container')
])


@app.callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value')
)
def update_output(value):
    return f'You have selected {value}'


if __name__ == '__main__':
    app.run_server(debug=True)
