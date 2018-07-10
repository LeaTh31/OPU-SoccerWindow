import Tkinter as tk
import time
import random
import os
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

def Draw():
    global score_color

    score_color = tk.Label(root, text='Initial text')
    score_color.pack(ipadx=50, ipady=50)

def get_values(line):
    split = line.split()
            
    cycle = int(split[1])
    
    current_values = []

    # Ball coordinates
    current_values.append(float(split[3]))
    current_values.append(float(split[4]))
			
    #Ball velocity
    current_values.append(float(split[5]))
    current_values.append(float(split[6][:-1]))

    # Players coordinates
    for j in range(len(split)):
        if split[j] == '((l' or split[j] == '((r':
            current_values.append(float(split[j+4]))
            current_values.append(float(split[j+5]))

            #Velocity
            current_values.append(float(split[j+6]))
            current_values.append(float(split[j+7]))
				
            current_values.append(float(split[j+8]))

            #Absolute neck angle
            current_values.append(float(float(split[j+8]) + float(split[j+9])))

        if split[j] == '(v':
            #View angle range
            current_values.append(float(split[j+2][:-1]))

    current_values = scaling.transform([current_values])
    return current_values, cycle


def Refresher():
    global score_color
    #global sscore

    where = file.tell()
    line = file.readline()
    if not line:
        time.sleep(0.01)
        file.seek(where)
    else:
        if ("show" in line):
            #start = time.time()
            current_values, cycle = get_values(line)
            #print("Time to read line: " + str(time.time()-start))
            #start = time.time()
            y = model.predict(current_values)
            #print("Time to predict sscore: " + str(time.time()-start))
            if y[0] < 100:
                sscore = int(round(y[0])) - 100
            else:
                sscore = int(round(y[0])) - 99
                if sscore > 100:
                    sscore = 100

            (R,G,B) = (255,255,255)
            if sscore > 0:
                R -= int(2.55*sscore)
                B -= int(2.55*sscore)
            else:
                G += int(2.55*sscore)
                B += int(2.55*sscore)

            color = '#%02x%02x%02x' % (R, G, B)
            print("Cycle " + str(cycle) + " Situation Score prediction: " + str(sscore) + "    " + str(y[0]))
            score_color.configure(text=str(sscore), background=color)

    root.after(40, Refresher) # ms



train_file = "alln_train.csv"
val_file = "alln_val.csv"

REBUILD_MODEL = False
MAXD = 20
NB_TREES = 100

if REBUILD_MODEL:
    print("Working with: " + train_file + " and " + val_file)
    train_csv = pd.read_csv(train_file)
    val_csv = pd.read_csv(val_file)
    train_set = train_csv.values
    val_set = val_csv.values
    NB_PARAM = len(train_set[0])
    x_train = train_set[:,1:NB_PARAM]
    y_train = train_set[:,0]
    x_test = val_set [0,1:NB_PARAM]
    x_val = val_set[:,1:NB_PARAM]
    y_val = val_set[:,0]
    N_val = len(val_set)

    scaling = MinMaxScaler(feature_range=(0,1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_val = scaling.transform(x_val)

    print("Number of trees: " + str(NB_TREES))
    print("Depth: " + str(MAXD))
    #model = ExtraTreesRegressor(n_estimators=NB_TREES,max_depth=MAXD)
    model = RandomForestRegressor(n_estimators=NB_TREES,max_depth=MAXD)
    start = time.time()
    model.fit(x_train,y_train)
    ttime = time.time()
    y = model.predict(x_val)
    ptime = time.time()
    diffs = [abs(y[i]-y_val[i]) for i in range(N_val)]
    avg_diff = sum(diffs)/N_val
    print("Average difference: " + str(avg_diff) + "  Training time: " + str(ttime-start) + "  Prediction time: " + str(ptime-ttime))
    start = time.time()
    y = model.predict([x_test])
    end = time.time()
    print("Time for a single estimation: " + str(end-start))
    joblib.dump(model, 'model.pkl', protocol=2) 
    print ("The model has been dumped, it should be OK")

else:
    # Need to define scaler
    val_csv = pd.read_csv(val_file)
    val_set = val_csv.values
    NB_PARAM = len(val_set[0])
    x_val = val_set[:,1:NB_PARAM]
    scaling = MinMaxScaler(feature_range=(0,1)).fit(x_val)
    start = time.time()
    model = joblib.load('model.pkl')
    print("Time to load existing model: " + str(time.time()-start))


filename = 'incomplete.rcg'
file = open(filename,'r')

#Find the size of the file and move to the end
st_results = os.stat(filename)
st_size = st_results[6]
file.seek(st_size)
sscore = 0

root=tk.Tk()
Draw()
Refresher()
root.mainloop()
