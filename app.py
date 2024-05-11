import gradio
import joblib
import numpy as np
import pandas as pd

age = gradio.Slider(1, 100,label="age",info = 'age')
anaemia = gradio.Radio(["True", "False"],label="anaemia",info = 'have anaemia')
creatinine_phosphokinase = gradio.Slider(1, 3000,label="creatine",info = 'creatine range')
diabetes = gradio.Radio(["True", "False"],label="diabetes",info = 'have diabetes')
ejection_fraction = gradio.Slider(1, 100,label="ejaection",info = 'ejection fraction')
high_blood_pressure = gradio.Radio(["True", "False"],label="high BP",info = 'is high BP')
platelets = gradio.Slider(1, 1000000,label="platelets",info = 'platelets count')
serum_creatinine = gradio.Slider(0.0, 12.0,label="serum creatine",info = 'serum creatine')
serum_sodium = gradio.Slider(0, 200,label="serum sodium",info = 'serum sodium')
sex = gradio.Radio(["True", "False"],label="sex",info = 'sex')
smoking = gradio.Radio(["True", "False"],label="smoking",info = 'smoking')
time = gradio.Slider(0, 365,label="time",info = 'followup days')

def predict_death_event(age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time):

    input_df = pd.DataFrame({"age": [int(age)], 
                            "anaemia": [bool(anaemia)], 
                            "creatinine_phosphokinase": [float(creatinine_phosphokinase)],
                            "diabetes": [bool(diabetes)], 
                            "ejection_fraction": [float(ejection_fraction)], 
                            "high_blood_pressure": [bool(high_blood_pressure)], 
                            "platelets": [float(platelets)], 
                            "serum_creatinine": [float(serum_creatinine)], 
                            "serum_sodium": [int(serum_sodium)], 
                            "sex": [bool(sex)], 
                            "smoking": [bool(smoking)], 
                            "time": [int(time)]})

    loaded_model = joblib.load('xgboost-model.pkl')    
    X_test = input_df

    y_pred = loaded_model.predict(X_test)
    predictions = [round(value) for value in y_pred]    

    return predictions


title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gradio.Interface(fn = predict_death_event,
                         inputs = [age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time],
                         outputs = gradio.Textbox(),
                         title = title,
                         description = description,
                         allow_flagging='never')

iface.launch(server_name="0.0.0.0", server_port = 8001)  # server_name="0.0.0.0", server_port = 8001   # Ref: https://www.gradio.app/docs/interface
