import pandas as pd
import numpy as np
import tensorflow as tf
from keras.src.layers import LSTM, Dropout, Bidirectional, Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.src.models import Sequential
from keras.src.utils import to_categorical
from keras._tf_keras.keras.layers import Embedding
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.optimizers import Adam
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
sns.set_style("darkgrid")

print(tf.__version__)

PATH = "C:/Users/16096/machine-learning-biotech/machine-learning-practice/ProteinClass/datasets/pfam/"

files =[]

df = pd.read_csv(PATH+f"pdb_data_seq.csv", index_col=None, header=0)
files.append(df)






#Checking Dataset


#Coheck completeness of data to summarize by column
#df.isna().sum()

#Look at model's output to check total number os instances by column and get top five most common entries
"""print(df["structureId"].groupby(df["structureId"]).value_counts().nlargest(5))"""

#180 entires is a fine cutoff
#seaborn to see mean and median seuqence lengths for model later
"""sns.displot(df["sequence"].apply(lambda x: len(x)), bins=75, height=4, aspect=2)
plt.show()"""

#Prep dataset for model by balancing and filter for classifications wit at least 180 observations
df_filt = df.groupby('structureId').filter(lambda x: len(x) > 180)
df_bal = df_filt.groupby('structureId').apply(lambda x: x.sample(180))

#Check number of classes with value_counts() -- 14
df_red = df_bal[['structureId', 'sequence']].reset_index(drop=True)
num_classes = len(df_red.structureId.value_counts())
print(num_classes)







#split data into train, test, and validation sets
X_train, X_test = train_test_split(df_red, test_size=0.25)
X_val, X_test = train_test_split(X_test, test_size=0.5)








#Preprocessing

#Reduce the sequences into the 20 most common amino acids and convert the sequences into integers for model speed
#Create dictionary
seq_dict = {'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20}

#Create a function to convert the string values into integers
def seq_encoder(data):
    full_list = []
    for i in data:
        row_seq = []
        for j in i:
            row_seq.append(seq_dict.get(j, 0))
        #TensorFlow expects a NumPy array so convert and add to full_list
        full_list.append(np.array(row_seq))
    return full_list

X_train_enc = seq_encoder(X_train)
X_val_enc = seq_encoder(X_val)
X_test_enc = seq_encoder(X_test)

#Pad sequences at end to ensure equal length
#truncate at end to ensure equal length. In the case of protein classification and LSTM we require equal length inputs. We are only focused on the common amino acids and can get rid of data in a sequence for the sake of the classification model.
X_train_pad = pad_sequences(X_train_enc, maxlen=100, padding='post', truncating='post')
X_val_pad = pad_sequences(X_val_enc, maxlen=100, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_enc, maxlen=100, padding='post', truncating='post')


#Preprocess OUTPUT data
LE = LabelEncoder()
y_train_enc = LE.fit_transform(X_train['structureId'])
y_val_enc = LE.transform(X_val['structureId'])
y_test_enc = LE.transform(X_test['structureId'])

#Turn class vector into binary matrix class matrix with to_categorical
y_train = to_categorical(y_train_enc)
y_val = to_categorical(y_val_enc)
y_test = to_categorical(y_test_enc)


#Developing the Model Architecture with Keras

#Start by making an instance of the sequential class and adding layers
#add embedding layer to convert positive integers into dense vectors. 
#Length of amino acid index + 1 =21
model = Sequential()
model.add(Embedding(input_dim=21, output_dim=32, name="EmbeddingL"))

#Add a LSTM layer, wrapped in a bidrectional layer, to run inputs in both past to future and future to past
model.add(Bidirectional(LSTM(8), name="BidirectionalL"))

#Add a dropout layer to prevent overfitting
model.add(Dropout(0.2, name="DropoutL"))

#Add a Dense layer and put 14 nodes for the possible classes
model.add(Dense(14, activation='softmax', name="DenseL"))



#Optimizer
optimizer = Adam(learning_rate=0.1)


#Compile
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



#train with a callback when model is no longer learning
from keras.src.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)


#mlflow
mlflow.keras.autolog()
history = model.fit(X_train_pad, y_train, epochs=30, batch_size=256, validation_data=(X_val_pad, y_val), callbacks=[earlystop])
