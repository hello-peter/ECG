import mxnet as mx
import pandas as pd
import numpy as np

#extract data function
def data_iter(train = True):
    #extract the '.arff' file
    def extract_arff(path):
        with open(path) as fp:
            con = fp.readlines()
        train_data = list()
        for record in con:
            a = list()
            value = record.split(',')
            for fringe in value:
                try:
                    a.append(float(fringe))
                except:
                    pass
            train_data.append(a)
        df = pd.DataFrame(train_data)
        label = df[df.shape[1] - 1]
        del df[df.shape[1] - 1]
        return df,label

     #extract the '.txt' file
    def extract_txt(path):
        with open(path) as fp:
            content = fp.readlines()
        con = list()
        for lenth in range(len(content)):
            line = list()
            col = content[lenth].split(' ')
            for value in col:
                try:
                    a = float(value)
                    line.append(a)
                except:
                    pass
            con.append(line)
        df = pd.DataFrame(con)
        label = df[0]
        del df[0]
        df = pd.DataFrame(np.array(df).tolist())
        return df,label

    def concat(df1 , df2):
        return df1.append(df2,ignore_index = True)

    #Determine whether is Train data iter
    if train == True:
        path_arff = './ECG5000/ECG5000_TRAIN.arff'
        path_txt = './ECG5000/ECG5000_TRAIN.txt'
    else:
        path_txt = './ECG5000/ECG5000_TEST.txt'
        path_arff = './ECG5000/ECG5000_TEST.arff'

    df_txt, y_txt = extract_txt(path = path_txt)
    df_arff,y_arff = extract_arff(path = path_arff)
    
    sample = mx.nd.array(concat(df_arff,df_txt))[:999].reshape(shape = (999,1,140))
    label = mx.nd.array(concat(y_arff,y_txt))[1:].reshape(shape = (999,1,))
    yield (sample.astype('float32'),label.astype('float32'))

