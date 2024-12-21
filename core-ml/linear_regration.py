import tensorflow as tf
import numpy as np
import pandas as pd


deftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #training data
defeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') #testing data

y_train = deftrain.pop('survived')
y_eval = defeval.pop('survived')

CATEGORICAL_COLUMN = [
    'sex','n_siblings_spouses','parch','class','deck','embark_town','alone'
]

NUMERICAL_COLUMN = [
    'age','fare'
]

feature_columns = []

for feature_name in CATEGORICAL_COLUMN:
    vocabulary = deftrain[feature_name].unique() # unique values 
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))
    print(feature_columns)

for feature_name in NUMERICAL_COLUMN:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))

print(feature_columns)        

def make_input_fn(data_df,label_df,shuffle=True,num_epochs=10,batch_size=32):
    def input_fn():
        ds = tf.data.Dataset.from_tensor_slices(dict(data_df),label_df)
        if shuffle:
            ds = ds.shuffle(1000)
            ds = ds.batch(batch_size).repeat(num_epochs)
            return ds
        return input_fn
    
train_input_fn = make_input_fn(deftrain,y_train)
eval_input_fn = make_input_fn(defeval,y_eval,num_epochs=1,shuffle=False)    

#mode creation using keras

feature_layer = tf.keras.layers.DanseFeatures(feature_columns)
model = tf.keras.Sequential(
    [
        feature_layer,
        tf.keras.layers.Dense(1,activation='sigmoid')
    ]
)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_input_fn)

