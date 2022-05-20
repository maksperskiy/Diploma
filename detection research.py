#!/usr/bin/env python
# coding: utf-8

# # 1.

# In[1]:


get_ipython().run_line_magic('pip', 'install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib')


# In[1]:


import cv2
import numpy as np

import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp


# In[ ]:





# # 2.

# In[2]:


mp_holistic = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# In[3]:


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# In[4]:


def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())


# In[5]:


def extract_keypoints(results):
    try:
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            lh = np.array([[res.x, res.y, res.z] for res in
                           results.multi_hand_landmarks[idx].landmark]).flatten() \
                if results.multi_handedness[idx].classification[0].label == 'Left' else np.zeros(21 * 3)
            rh = np.array([[res.x, res.y, res.z] for res in
                           results.multi_hand_landmarks[idx].landmark]).flatten() \
                if results.multi_handedness[idx].classification[0].label == 'Right' else np.zeros(21 * 3)
        return np.concatenate([lh, rh])
    except:
        return np.concatenate([np.zeros(21 * 3), np.zeros(21 * 3)])


# # 3.

# In[6]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[7]:


actions = np.array([ 'up', 'down', 'left', 'right', 'ok', 'back'])


# In[8]:


label_map = {label:num for num, label in enumerate(actions)}


# In[9]:


label_map


# In[11]:


sequences = np.load("data/sequences.npy")
labels = np.load("data/labels.npy")


# In[12]:


def interp_coords(x):
    coords = []
    for num,i in enumerate(x):
        if np.count_nonzero(i) != 0:
            coords.append([i,num])
    if not coords:
        return x
    result = []
    for i in range(126):
        result.append(np.interp(range(40),[e[1] for e in coords],[e[0][i] for e in coords]))
    return np.array(result).transpose()


# In[13]:


sequences = np.array([interp_coords(e) for e in sequences])


# In[14]:


np.array(sequences).shape


# In[15]:


sequences[117,:,13]


# In[16]:


np.array(labels).shape


# In[17]:


X = np.array(sequences)


# In[18]:


X.shape


# In[19]:


y = to_categorical(labels).astype(int)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# In[21]:


y_test.shape


# # 4.

# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


# In[23]:


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


# In[38]:


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(40,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


# In[39]:


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[42]:


model.summary()


# In[27]:


model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])


# In[28]:


model.summary()


# # 5.

# In[29]:


res = model.predict(X_test)


# In[30]:


actions[np.argmax(res[4])]


# In[31]:


actions[np.argmax(y_test[4])]


# # 6. 

# In[33]:


model.save('models/gesture.h5')


# In[32]:


del model


# In[24]:


model.load_weights('models/action.h5')


# # 7. 

# In[32]:


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, recall_score


# In[33]:


yhat = model.predict(X_test)


# In[34]:


ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


# In[35]:


multilabel_confusion_matrix(ytrue, yhat)


# In[36]:


accuracy_score(ytrue, yhat)


# # 8.

# In[39]:


from scipy import stats


# In[40]:


colors = [(245,117,16), (117,245,16), (16,117,245), (16,217,245), (116,117,245), (116,217,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


# In[41]:


sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
with mp_holistic.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        draw_landmarks(image, results)
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-40:]
        
        if len(sequence) == 40:
            res = model.predict(np.expand_dims(interp_coords(sequence), axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

