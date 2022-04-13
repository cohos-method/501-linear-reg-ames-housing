import dash
from dash import dcc,html
from dash.dependencies import Input, Output, State
import pickle
from sklearn import datasets, linear_model, metrics


########### Define your variables ######
myheading1='Predicting Heart Disease'
image1='image1.jpeg'
tabtitle = 'Predicting Heart Disease'
sourceurl = 'https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease'
githublink = 'https://github.com/cohos-method/501-linear-reg-ames-housing'
AgeCatList= ['55-59', '80 or older', '65-69', '75-79', '40-44', '70-74', '60-64', '50-54', '45-49', '18-24', '35-39', '30-34', '25-29']
AgeCatList.sort()
RaceCatList= ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic']
RaceCatList.sort()
GenHealthCatList= ['Very good', 'Fair', 'Good', 'Poor', 'Excellent']
GenHealthCatList.sort()

def buildOptionsDict(lbl):
    opt = []
    for i in range(len(lbl)):
        d = {}
        d['label'] = lbl[i]
        d['value'] = lbl[i]
        opt.append(d)
    return opt

def buildFeatures(BMI=28.32 # Number
         , Smoking='No' #Yes/No
         , AlcoholDrinking='No' #Yes/No
         , Stroke='No' #Yes/No
         , PhysicalHealth=3.3 #Range
         , MentalHealth=3.8 #Range
         , DiffWalking='No' #Yes/No
         , Sex='Male' #Male/Female
         , AgeCategory='40-44' #Number
         , Race='Other' #Category
         , Diabetic='No' #Yes/No
         , PhysicalActivity='No' #Yes/No
         , GenHealth=15 #number
         , SleepTime=8 #number
         , Asthma='No' #Yes/No
         , KidneyDisease='No' #Yes/No
         , SkinCancer='No'): #Yes/No

    YesNo2Binary = lambda flg: 1 if flg == 'Yes' else 0
    Sex2Binary = lambda flg: 1 if flg == 'Male' else 0
    prefixer = lambda l, prefix: [ prefix + c for c in l]
    dataListBuilder = lambda l, data: [ 1 if c == data else 0 for c in l]
    buildDict = lambda keys, values: dict(zip(keys, values))

    AgeCatList= ['55-59', '80 or older', '65-69', '75-79', '40-44', '70-74', '60-64', '50-54', '45-49', '18-24', '35-39', '30-34', '25-29']
    AgeCatList.sort()
    AgeCatColPrefix = 'AgeCategory_'
    AgeCatCols = prefixer(AgeCatList, AgeCatColPrefix)
    AgeCatColsData = dataListBuilder (AgeCatList, AgeCategory)
    AgeCatData = buildDict(AgeCatCols, AgeCatColsData)
    print ("\n*** AgeCatData", AgeCatData)

    RaceCatList= ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic']
    RaceCatList.sort()
    RaceCatColPrefix = 'Race_'
    RaceCatCols = prefixer(RaceCatList, RaceCatColPrefix)
    RaceCatColsData = dataListBuilder (RaceCatList, Race)
    RaceCatData = buildDict(RaceCatCols, RaceCatColsData)
    print ("\n*** RaceCatData", RaceCatData)

    GenHealthCatList= ['Very good', 'Fair', 'Good', 'Poor', 'Excellent']
    GenHealthCatList.sort()
    GenHealthCatColPrefix = 'GenHealth_'
    GenHealthCatCols = prefixer(GenHealthCatList, GenHealthCatColPrefix)
    GenHealthCatColsData = dataListBuilder (GenHealthCatList, GenHealth)
    GenHealthData = buildDict(GenHealthCatCols, GenHealthCatColsData)
    print ("\n*** GenHealthData", GenHealthData)

    PhysicalHealthMin= 0.0
    PhysicalHealthMax= 30.0
    MentalHealthMin= 0.0
    MentalHealthMax= 30.0
    SleepTimeMin= 1.0
    SleepTimeMax= 24.0

    d = {'BMI' : BMI
         , 'Smoking' : YesNo2Binary(Smoking)
         , 'AlcoholDrinking' : YesNo2Binary(AlcoholDrinking)
         , 'Stroke' : YesNo2Binary(Stroke)
         , 'PhysicalHealth' : PhysicalHealth
         , 'MentalHealth' : MentalHealth
         , 'DiffWalking': YesNo2Binary(DiffWalking)
         , 'Sex' : Sex2Binary(Sex)
         , 'Diabetic' : YesNo2Binary(Diabetic)
         , 'PhysicalActivity' : YesNo2Binary(PhysicalActivity)
         , 'SleepTime' : SleepTime
         , 'Asthma' : YesNo2Binary(Asthma)
         , 'KidneyDisease' : YesNo2Binary(KidneyDisease)
         , 'SkinCancer' : YesNo2Binary(SkinCancer)
         , 'AgeCategory_18-24' : AgeCatData['AgeCategory_18-24']
         , 'AgeCategory_25-29' : AgeCatData['AgeCategory_25-29']
         , 'AgeCategory_30-34' : AgeCatData['AgeCategory_30-34']
         , 'AgeCategory_35-39' : AgeCatData['AgeCategory_35-39']
         , 'AgeCategory_40-44' : AgeCatData['AgeCategory_40-44']
         , 'AgeCategory_45-49' : AgeCatData['AgeCategory_45-49']
         , 'AgeCategory_50-54' : AgeCatData['AgeCategory_50-54']
         , 'AgeCategory_55-59' : AgeCatData['AgeCategory_55-59']
         , 'AgeCategory_60-64' : AgeCatData['AgeCategory_60-64']
         , 'AgeCategory_65-69' : AgeCatData['AgeCategory_65-69']
         , 'AgeCategory_70-74' : AgeCatData['AgeCategory_70-74']
         , 'AgeCategory_75-79' : AgeCatData['AgeCategory_75-79']
         , 'AgeCategory_80 or older' : AgeCatData['AgeCategory_80 or older']
         , 'Race_American Indian/Alaskan Native' : RaceCatData['Race_American Indian/Alaskan Native']
         , 'Race_Asian' : RaceCatData['Race_Asian']
         , 'Race_Black' : RaceCatData['Race_Black']
         , 'Race_Hispanic' : RaceCatData['Race_Hispanic']
         , 'Race_Other' : RaceCatData['Race_Other']
         , 'Race_White' : RaceCatData['Race_White']
         , 'GenHealth_Excellent' : GenHealthData['GenHealth_Excellent']
         , 'GenHealth_Fair' : GenHealthData['GenHealth_Excellent']
         , 'GenHealth_Good' : GenHealthData['GenHealth_Good']
         , 'GenHealth_Poor' : GenHealthData['GenHealth_Poor']
         , 'GenHealth_Very good':GenHealthData['GenHealth_Very good']}
    print ("\n\n*** d", d)
    return d

###init options
AgeOpt = buildOptionsDict(AgeCatList)
RaceOpt = buildOptionsDict(RaceCatList)
GenHealthOpt = buildOptionsDict(GenHealthCatList)
SexOpt = buildOptionsDict(['Male', 'Female'])
YesNoOpt = buildOptionsDict(['Yes', 'No'])
print(AgeOpt, RaceOpt, GenHealthOpt, SexOpt, YesNoOpt)
#### load model
filename = open('cohos_lr_model.pkl', 'rb')
unpickled_model = pickle.load(filename)
filename.close()


########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle
#BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer
########### Set up the layout
app.layout = html.Div(children=[
    html.H1(myheading1),
    html.Div([
        html.Img(src=app.get_asset_url(image1), style={'width': '30%', 'height': 'auto'}, className='four columns'),
        html.Div([
                html.H3('Features of Heart Health:'),
                html.Div('BMI:'),
                dcc.Slider(0, 100, 1,value=28,id='BMI',tooltip={"placement": "bottom", "always_visible": True}), ##{0: '0',10: '10',20: '20',30: '30',40: '40',50: '50',60: '60',70: '70',80: '80',90: '90',100: '100'}),
                html.Br(),
                html.Div('Smoking:'),
                dcc.RadioItems(options=YesNoOpt, value='No', inline=False, id='Smoking'),
                html.Br(),
                html.Div('AlcoholDrinking:'),
                dcc.RadioItems(options=YesNoOpt, value='No', inline=True, id='AlcoholDrinking'),
                html.Br(),
                html.Div('Stroke:'),
                dcc.RadioItems(options=YesNoOpt, value='No', inline=True, id='Stroke'),
                html.Br(),
                html.Div('PhysicalHealth:'),
                dcc.Slider(0, 100, 1,value=28,id='PhysicalHealth',tooltip={"placement": "bottom", "always_visible": True}), ##{0: '0',10: '10',20: '20',30: '30',40: '40',50: '50',60: '60',70: '70',80: '80',90: '90',100: '100'}),
                html.Br(),
                html.Br(),
                html.Div('MentalHealth:'),
                dcc.Slider(0, 100, 1,value=28,id='MentalHealth',tooltip={"placement": "bottom", "always_visible": True}), ##{0: '0',10: '10',20: '20',30: '30',40: '40',50: '50',60: '60',70: '70',80: '80',90: '90',100: '100'}),
                html.Br(),
                html.Div('DiffWalking:'),
                dcc.RadioItems(options=YesNoOpt, value='No', inline=True, id='DiffWalking'),
                html.Br(),
                html.Div('Sex:'),
                dcc.RadioItems(options=SexOpt, value='Male', inline=True, id='Sex'),
                html.Br(),
                html.Div('AgeCategory:'),
                dcc.Dropdown(options=AgeOpt, value='18-24', id='AgeCategory'),
                html.Br(),
                html.Div('Race:'),
                dcc.Dropdown(options=RaceOpt, value='White', id='Race'),
                html.Br(),
                html.Div('Diabetic:'),
                dcc.RadioItems(options=YesNoOpt, value='No', inline=True, id='Diabetic'),
                html.Br(),
                html.Div('PhysicalActivity:'),
                dcc.RadioItems(options=YesNoOpt, value='No', inline=True, id='PhysicalActivity'),
                html.Br(),
                html.Div('GenHealth:'),
                dcc.Dropdown(options=GenHealthOpt, value='Good', id='GenHealth'),
                html.Br(),
                html.Div('SleepTime:'),
                dcc.Slider(0, 24, .5,value=8,id='SleepTime',tooltip={"placement": "top", "always_visible": True}), ##{0: '0 hrs',2: '2 hrs',4: '4 hrs',6: '6 hrs',8: '8 hrs',10: '10 hrs',12: '12 hrs',14: '14 hrs',16: '16 hrs',18: '18 hrs',20: '20 hrs'}),
                html.Br(),
                html.Div('Asthma:'),
                dcc.RadioItems(options=YesNoOpt, value='No', inline=True, id='Asthma'),
                html.Br(),
                html.Div('KidneyDisease:'),
                dcc.RadioItems(options=YesNoOpt, value='No', inline=True, id='KidneyDisease'),
                html.Br(),
                html.Div('SkinCancer:'),
                dcc.RadioItems(options=YesNoOpt, value='No', inline=True, id='SkinCancer'),
            ], className='four columns'),
            html.Div([
                html.Button(children='Submit', id='submit-val', n_clicks=0,
                                style={
                                'background-color': 'red',
                                'color': 'white',
                                'margin-left': '5px',
                                'verticalAlign': 'center',
                                'horizontalAlign': 'center'}
                                ),
                #html.H3('Predicted Heart Disease:'),
                html.H3(id='Results'),
                html.Br(),
                html.H4('Regression Equation:'),
                html.Label('Predict = 1296682353.2724  + (0.0003 * BMI) + (0.0226 * Smoking) + (-0.0119 * AlcoholDrinking) + (0.1782 * Stroke) + (0.0005 * PhysicalHealth) + (-0.0 * MentalHealth) + (0.0331 * DiffWalking) + (0.0455 * Sex) + (0.0621 * Diabetic) + (-0.0011 * PhysicalActivity) + (-0.0005 * SleepTime) + (0.0151 * Asthma) + (0.0878 * KidneyDisease) + (0.017 * SkinCancer) + (523831052.6438 * AgeCategory_18-24) + (523831052.641 * AgeCategory_25-29) + (523831052.6399 * AgeCategory_30-34) + (523831052.6379 * AgeCategory_35-39) + (523831052.6408 * AgeCategory_40-44) + (523831052.6464 * AgeCategory_45-49) + (523831052.6591 * AgeCategory_50-54) + (523831052.6663 * AgeCategory_55-59) + (523831052.687 * AgeCategory_60-64) + (523831052.7064 * AgeCategory_65-69) + (523831052.7347 * AgeCategory_70-74) + (523831052.7546 * AgeCategory_75-79) + (523831052.7868 * AgeCategory_80 or older) + (191161924.6074 * Race_American Indian/Alaskan Native) + (191161924.5933 * Race_Asian) + (191161924.5874 * Race_Black) + (191161924.5928 * Race_Hispanic) + (191161924.6042 * Race_Other) + (191161924.6044 * Race_White) + (-2011675330.5636 * GenHealth_Excellent) + (-2011675330.471 * GenHealth_Fair) + (-2011675330.5285 * GenHealth_Good) + (-2011675330.38 * GenHealth_Poor) + (-2011675330.5573 * GenHealth_Very good)'),
            ], className='four columns')
        ], className='twelve columns',
    ),
    html.Br(),
    html.Br(),
    #html.Br(),
    #html.H4('Regression Equation:'),
    #html.Div('Predict = 1296682353.2724  + (0.0003 * BMI) + (0.0226 * Smoking) + (-0.0119 * AlcoholDrinking) + (0.1782 * Stroke) + (0.0005 * PhysicalHealth) + (-0.0 * MentalHealth) + (0.0331 * DiffWalking) + (0.0455 * Sex) + (0.0621 * Diabetic) + (-0.0011 * PhysicalActivity) + (-0.0005 * SleepTime) + (0.0151 * Asthma) + (0.0878 * KidneyDisease) + (0.017 * SkinCancer) + (523831052.6438 * AgeCategory_18-24) + (523831052.641 * AgeCategory_25-29) + (523831052.6399 * AgeCategory_30-34) + (523831052.6379 * AgeCategory_35-39) + (523831052.6408 * AgeCategory_40-44) + (523831052.6464 * AgeCategory_45-49) + (523831052.6591 * AgeCategory_50-54) + (523831052.6663 * AgeCategory_55-59) + (523831052.687 * AgeCategory_60-64) + (523831052.7064 * AgeCategory_65-69) + (523831052.7347 * AgeCategory_70-74) + (523831052.7546 * AgeCategory_75-79) + (523831052.7868 * AgeCategory_80 or older) + (191161924.6074 * Race_American Indian/Alaskan Native) + (191161924.5933 * Race_Asian) + (191161924.5874 * Race_Black) + (191161924.5928 * Race_Hispanic) + (191161924.6042 * Race_Other) + (191161924.6044 * Race_White) + (-2011675330.5636 * GenHealth_Excellent) + (-2011675330.471 * GenHealth_Fair) + (-2011675330.5285 * GenHealth_Good) + (-2011675330.38 * GenHealth_Poor) + (-2011675330.5573 * GenHealth_Very good)'),
    #html.Br(),
    html.A('Code on Github', href=githublink),
    html.Br(),
    html.A("Data Source", href=sourceurl),
    ]
)

#BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking',
#'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime',
#'Asthma', 'KidneyDisease', 'SkinCancer

######### Define Callback
@app.callback(
    Output(component_id='Results', component_property='children'),
    Input(component_id='submit-val', component_property='n_clicks'),
    State(component_id='BMI', component_property='value'),
    State(component_id='Smoking', component_property='value'),
    State(component_id='AlcoholDrinking', component_property='value'),
    State(component_id='Stroke', component_property='value'),
    State(component_id='PhysicalHealth', component_property='value'),
    State(component_id='MentalHealth', component_property='value'),
    State(component_id='DiffWalking', component_property='value'),
    State(component_id='Sex', component_property='value'),
    State(component_id='AgeCategory', component_property='value'),
    State(component_id='Race', component_property='value'),
    State(component_id='Diabetic', component_property='value'),
    State(component_id='PhysicalActivity', component_property='value'),
    State(component_id='GenHealth', component_property='value'),
    State(component_id='SleepTime', component_property='value'),
    State(component_id='Asthma', component_property='value'),
    State(component_id='KidneyDisease', component_property='value'),
    State(component_id='SkinCancer', component_property='value'),

)
def ames_lr_function(clicks
                    , BMI
                    , Smoking
                    , AlcoholDrinking
                    , Stroke
                    , PhysicalHealth
                    , MentalHealth
                    , DiffWalking
                    , Sex
                    , AgeCategory
                    , Race
                    , Diabetic
                    , PhysicalActivity
                    , GenHealth
                    , SleepTime
                    , Asthma
                    , KidneyDisease
                    , SkinCancer
                    ):
    if clicks==0:
        return "Input the data and Submit"
    else:
        dic = buildFeatures(BMI
                            , Smoking
                            , AlcoholDrinking
                            , Stroke
                            , PhysicalHealth
                            , MentalHealth
                            , DiffWalking
                            , Sex
                            , AgeCategory
                            , Race
                            , Diabetic
                            , PhysicalActivity
                            , GenHealth
                            , SleepTime
                            , Asthma
                            , KidneyDisease
                            , SkinCancer)
        fts = [list(dic.values())]
        ##res = unpickled_model.predict(fts)
        #formatted_y = "${:,.2f}".format(y[0])
        ## return "Predicted Heart Disease: " + str(round(res[0] * 100, 2)) + "%"
        return fts



############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
