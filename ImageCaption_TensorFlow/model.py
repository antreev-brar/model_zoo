import keras
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras import Input, layers
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense,Input , Dropout , RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.layers.merge import add
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
def data_generator(descriptions , photos , wordtoix , max_length , num_photos_per_batch):
  x1,x2 ,y = list(), list() , list()
  n = 0
  while( True):
    for key , desc_list in descriptions.items():
      n+=1 
      photo = photos['/'+key+'.jpg']
      for desc in desc_list:
        seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]

        for i in range(1 , len(seq )):
          in_seq ,out_seq = seq[:i] , seq[i]
          in_seq  = pad_sequences([in_seq], maxlen = max_length)[0]
          out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]
          
          x1.append(photo)
          x2.append(in_seq)
          y.append(out_seq)

        if n == num_photos_per_batch : 
          yield [[array(x1) , array(x2) ],array(y)]
          x1,x2 ,y = list(), list() , list()
          n= 0


def inception_model():
    
    model = InceptionV3(weights = 'imagenet')
    model.summary()

    model_new = Model(model.inputs , model.layers[-2].output)
    print(model_new.summary())
    return model_new



