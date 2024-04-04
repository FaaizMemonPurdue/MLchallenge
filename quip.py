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
from constants import name_strind, name_ind, bads
from typing import List, Dict, Any
import cv2 
import pickle

import sys
from csvreader import writeGuessCSV

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
            
def extract_ffold(src_folder:str, fpkl):    
    if src_folder.endswith("/"):
        src_folder = src_folder[:-1]
    
    max_faces = len(os.listdir(src_folder))
    face_data = np.empty((max_faces, 224, 224, 3))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    for imno in range(max_faces):
        fname = f"{imno}.jpg"
        if fname[:-4] in bads:
            continue

        img_path = os.path.join(src_folder, fname)
        if not os.path.exists(img_path):
            print(f"{img_path} dne, stopping")
            break

        try:
            source_objs = detection.extract_faces(
                img_path=img_path,
                target_size=(224, 224),
                detector_backend="mtcnn",
                human_readable=False, #i want RGB out here bc normalize flips to BGR
                grayscale=False,
                enforce_detection=False,
                align=True,
                expand_percentage=0,
            )
        except ValueError as e:
            print(f"{e} on {fname}")
            continue

        chosen_face = face_pickin(source_objs)["face"]
        # print(chosen_face.shape)
        # cv2.imwrite("com.jpg", chosen_face)
        # norm = normalize(chosen_face, "channels_last")
        norm = chosen_face
        face_data[imno] = norm
        # if used > 2:
        #     break

    # Pickle the data
    with open(f"{fpkl}.pkl", 'wb') as f:
        pickle.dump((face_data), f)
    print(face_data.shape)
    return face_data

def extract_labels(label_path, lpkl):
    labels = pd.read_csv(label_path)
    max_faces = len(labels)
    label_int = np.array([name_ind[row['Category']] for _, row in labels.iterrows()])
    with open(f"{lpkl}.pkl", 'wb') as f:
        pickle.dump((label_int), f)
    return label_int

def retrieve_ffold(fpkl):
    with open(f"{fpkl}.pkl", 'rb') as f:
        face_data = pickle.load(f)
    return face_data

def retrieve_labels(lpkl):
    with open(f"{lpkl}.pkl", 'rb') as f:
        label_data = pickle.load(f)
    return label_data

# def train_vgg(fpkl, lpkl, h5_out, hard:bool = False):
#     X_train, y_train = retrieve_ffold(fpkl), retrieve_labels(lpkl)
    
#     # Load VGGFace model with pre-trained weights
#     tf_session = tf.compat.v1.Session()
#     tf.compat.v1.keras.backend.set_session(tf_session)
    
#     vggface_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), weights='vggface')

#     # Freeze the layers in the pre-trained model
#     for layer in vggface_model.layers:
#         layer.trainable = False

#     # Flatten the output of the last convolutional layer
#     num_classes = 100  # Example: if you have 100 classes
#     hidden_dim = 4096

#     last_layer = vggface_model.get_layer('pool5').output
#     x = Flatten(name='flatten')(last_layer)
#     x = Dense(hidden_dim, activation='relu', name='fc6')(x)
#     x = Dense(hidden_dim, activation='relu', name='fc7')(x)
#     predictions = Dense(num_classes, activation='softmax')(x)

#     # Create a new model
#     model = Model(inputs=vggface_model.input, outputs=predictions)

#     # Compile the model
#     model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#     model.summary()
#     # Checkpoint to save the model weights when validation accuracy improves
#     checkpoint = ModelCheckpoint(h5_out, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

#     # Train the model
#     model.fit(X_train, tku.to_categorical(y_train, num_classes=num_classes), 
#             batch_size=128, epochs=100, validation_split=0.2, callbacks=[checkpoint])

#     # After training, the best model (based on validation accuracy) will be saved to 'best_model.h5'


def train(X_train, y_train, h5_out, m="resnet50"):
    # X_train, y_train = retrieve_ffold(fpkl), retrieve_labels(lpkl)
    
    # Load VGGFace model with pre-trained weights
    tf_session = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(tf_session)
    if m == "resnet50":
        vggface_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), weights='vggface')

        # Freeze the layers in the pre-trained model
        for layer in vggface_model.layers:
            layer.trainable = False

        # Flatten the output of the last convolutional layer
        num_classes = 100  # Example: if you have 100 classes
        last_layer = vggface_model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        predictions = Dense(num_classes, activation='softmax', name='classifier')(x)
    else:
        vggface_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), weights='vggface')

        # Freeze the layers in the pre-trained model
        for layer in vggface_model.layers:
            layer.trainable = False

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
    
def tr_ts(fpkl, lpkl, h5_out, m="resnet50"):
    faces, labels = retrieve_ffold(fpkl), retrieve_labels(lpkl)
    faces_train, faces_test, labels_train, labels_test = train_test_split(faces, labels, test_size=0.5, random_state=42)
    train(faces_train, labels_train, h5_out, m)
    qtest(faces_test, labels_test, h5_out)
    # if not os.path.isdir("odir"):


# def test_blind(fpkl):

    #will need to output writer from preds
def qtest(X_test, y_test, h5_path, m="resnet50"):
    if m == "resnet50":
        vggface_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), weights='vggface')

        # Freeze the layers in the pre-trained model
        for layer in vggface_model.layers:
            layer.trainable = False

        # Flatten the output of the last convolutional layer
        num_classes = 100  # Example: if you have 100 classes
        last_layer = vggface_model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        predictions = Dense(num_classes, activation='softmax', name='classifier')(x)
    else:
        vggface_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), weights='vggface')

        # Freeze the layers in the pre-trained model
        for layer in vggface_model.layers:
            layer.trainable = False

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
    print(predicted_labels.shape)
    # Now you can compare the predicted labels to the actual labels to calculate the accuracy
    accuracy = np.mean(predicted_labels == y_test)

    print(f'Test accuracy: {accuracy * 100:.2f}%')

def uktest(fpkl, h5_path, m="resnet50"):
    X_test = retrieve_ffold(fpkl)
    if m == "resnet50":
        vggface_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), weights='vggface')

        # Freeze the layers in the pre-trained model
        for layer in vggface_model.layers:
            layer.trainable = False

        # Flatten the output of the last convolutional layer
        num_classes = 100  # Example: if you have 100 classes
        last_layer = vggface_model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        predictions = Dense(num_classes, activation='softmax', name='classifier')(x)
    else:
        vggface_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), weights='vggface')

        # Freeze the layers in the pre-trained model
        for layer in vggface_model.layers:
            layer.trainable = False

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
    print(predicted_labels.shape)
    # Now you can compare the predicted labels to the actual labels to calculate the accuracy
    writeGuessCSV(predicted_labels, "apr4.csv")
    



if __name__ == '__main__':
    # csv = str(sys.argv[1])
    # fpkl = str(sys.argv[2])
    # # fold = "../data/train_small_f"
    # # fpkl = "train_small_f_1"
    # extract_labels(csv, fpkl)    
    extract_ffold("../data/train_f", "train_full_f1")
    tr_ts("train_full_f1", "tr_labels", "tr_full_f1.h5")
    uktest("test_f1", "tr_full_f1.h5")

    # uktest("train_small_f1", "small_froze_res.h5")
