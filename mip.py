import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import tensorflow.keras.utils as tku
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from deepface.modules import detection, verification
# from deepface.modules import 
import os
import pandas as pd
from constants import name_strind, bads
from typing import List, Dict, Any
import cv2 
import pickle

def normalize(x, data_format): #once per img
    x_temp = np.copy(x)
    assert data_format in {'channels_last'}

    # x_temp = x_temp[..., ::-1] #flip rgb -> bgr

    mean_values = np.mean(x_temp, axis=(0, 1, 2))

    # Extract the mean value for each channel
    mean_blue = mean_values
    mean_green = mean_values
    mean_red = mean_values
    x_temp[..., 0] -= mean_blue
    x_temp[..., 1] -= mean_green
    x_temp[..., 2] -= mean_red

    return x_temp


def face_pickin(source_objs: List[Dict[str, Any]]):
    #ideally, if a lot pick high conf, if low check all the pairwise for repeats,
    if len(source_objs) == 1:
        return source_objs[0]
    else:
        mc = 0
        maxObj = None
        for obj in source_objs:
            if obj["confidence"] > mc:
                maxObj = obj
                mc = obj["confidence"]
        return maxObj
            

def extract_data(src_folder:str, label_path:str, train_part=0.9):    
    if src_folder.endswith("/"):
        src_folder = src_folder[:-1]
    labels = pd.read_csv(label_path)
    max_faces = len(labels)
    label_data = np.empty(max_faces)
    face_data = np.empty((max_faces, 224, 224, 3))
    used = 0
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    for index, row in labels.iterrows():
        fname = row['File Name']
        if fname[:-4] in bads:
            continue

        img_path = os.path.join(src_folder, fname)
        if not os.path.exists(img_path):
            print(f"{img_path} dne, stopping")
            break
        
        cname = row['Category']
        cnum = name_strind[cname]

        try:
            source_objs = detection.extract_faces(
                img_path=img_path,
                target_size=(224, 224),
                detector_backend="mtcnn",
                human_readable=False, #i want RGB out here bc preproc flips to BGR
                grayscale=False,
                enforce_detection=True,
                align=True,
                expand_percentage=0,
            )
        except ValueError as e:
            print(f"{e} on {fname}")
            continue

        chosen_face = face_pickin(source_objs)["face"]
        # print(chosen_face.shape)
        label_data[used] = cnum
        # cv2.imwrite("com.jpg", chosen_face[0] * 255)
        norm = normalize(chosen_face, "channels_last")
        face_data[used] = norm
        used += 1
        # if used > 2:
        #     break

    label_data = label_data[:used]
    face_data = face_data[:used]

    # Pickle the data
    face_train, face_test, label_train, label_test = train_test_split(face_data, label_data, train_size=train_part, random_state=1)
    with open(f"{os.path.basename(src_folder)}.pkl", 'wb') as f:
        pickle.dump((face_train, face_test, label_train, label_test), f)
    print(face_train.shape, label_train.shape)
    return face_train, face_test, label_train, label_test

def retrieve_data(sec, pkl_path="data.pkl"):
    with open(pkl_path, 'rb') as f:
        if sec == "train":
            face_data, _, label_data, _  = pickle.load(f)
        else:
            _, face_data, _, label_data  = pickle.load(f)
    print(face_data.shape, label_data.shape)
    return face_data, label_data 



def train(pkl_in, h5_out, hard:bool = False):
    X_train, y_train = retrieve_data("train", pkl_in)
    
    # Load VGGFace model with pre-trained weights
    tf_session = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(tf_session)

    vggface_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), weights='vggface')

    # Freeze the layers in the pre-trained model
    # for layer in vggface_model.layers:
    #     layer.trainable = False


    # Flatten the output of the last convolutional layer
    num_classes = 100  # Example: if you have 100 classes
    hidden_dim = 4096

    last_layer = vggface_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create a new model
    model = Model(inputs=vggface_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    # Checkpoint to save the model weights when validation accuracy improves
    checkpoint = ModelCheckpoint(h5_out, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Train the model
    model.fit(X_train, tku.to_categorical(y_train, num_classes=num_classes), 
            batch_size=128, epochs=100, validation_split=0.2, callbacks=[checkpoint])

    # After training, the best model (based on validation accuracy) will be saved to 'best_model.h5'

def test(src_folder:str, label_path, pkl_path, h5_path):
    X_test, y_test = retrieve_data("test", pkl_path)

    vggface_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), weights='vggface')
    for layer in vggface_model.layers:
        layer.trainable = False

    # Load the weights into the base model
    # vggface_model.load_weights('hss/small_1.h5')  # replace with the actual path

    # Define the structure of your custom model (should be the same as the one you trained)
    # Flatten the output of the last convolutional layer
    num_classes = 100  # Example: if you have 100 classes
    hidden_dim = 4096

    last_layer = vggface_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    # Create a new model
    model = Model(inputs=vggface_model.input, outputs=predictions)

    # Load the weights from the best model into your custom model
    model.load_weights(h5_path)
    
    # Use the model to predict the outputs for the test data
    predictions = model.predict(X_test)

    # The predictions are usually in the form of probabilities, so you might want to convert them to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Now you can compare the predicted labels to the actual labels to calculate the accuracy
    accuracy = np.mean(predicted_labels == y_test)

    print(f'Test accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    # extract_data("../data/train_small", "purdue-face-recognition-challenge-2024/train_small.csv")
    # train("../data/train_small", "purdue-face-recognition-challenge-2024/train_small.csv", hard=False)
    # test("../data/train_small", "purdue-face-recognition-challenge-2024/train_small.csv",
    #      "train_small.pkl", "h5s/small_1.h5")
    train("train_small.pkl", "h5s/small_2.h5")
    
    # X_train, y_train = retrieve_data("test", "train_small.pkl")
    # print(y_train.shape)