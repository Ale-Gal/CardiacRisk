import pandas as pd 
from re import findall
from pickle import dump
from timeit import default_timer as timer

from confidence import within_confidence_region, explain_negative
from grace2 import grace
from results_online_verification import print_results


if __name__=='__main__':
    
    #define the conditions
    event_threshold=40 #days 
    
    #load data
    #file only has relevant data -> feature columns and Event days
    raw_data = pd.read_csv('prova2.csv', sep=';')    
    
    #handle missing data -> substitute NaN for the median of the feature
    data = raw_data.fillna(raw_data.median().to_dict())
    
    #change event days to event class -> applying a mask to convert into a Binary classification problem
    data['Event'] = (data['Event'] < event_threshold).astype(int)
    
    #drop KILLIP or HF(signs) because they have .94 correlation
    data= data.drop(columns='HF (signs)')
    
    #X and y
    X_data= data.drop(columns='Event').values.tolist()
    y_data= data['Event'].tolist() #also works .to_numpy()
    print(sum(y_data))
    
    #X_labels=['SEX','Age','Enrl','RF','CCS>II','DEP ST','SBP','HR','KILLIP','TN','Creat','CAA','AAS','Angina','Kn. CAD']
    X_labels=['SEX','Age','Enrl','Diabetes','Hypercholest','Hyperten','Smoking','CCS>II','DEP ST','SBP','HR','KILLIP','TN','Creat','CAA','AAS','Angina', 'MI', 'MR', 'PTCA', 'CABG'] #classi 'RF' e 'Kn. CAD' rimpiazzate dai singoli val che le costituiscono; 'DEP ST'=Depression Segment mappa 'ST SegmDev'; 'AAS'=Acute Aortic Syndrome mappa 'Aspirina'.
       
    model=grace
    

    t=[]


    #run for each patient
    for patient in range(13), y in zip(X_data,y_data):    #range(n) dove n sta per il num di pz che ho
        
        #apply rs
        risk = model(patient,X_labels)
        t.append(int(risk))
        
return t
         