import cv2
import numpy as np
import mediapipe as mp
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from keras.models import load_model
import pyautogui as pag

actions = np.array([ 'up', 'down', 'left', 'right', 'ok', 'back'])
key_actions = {'up': 'up', 
                'down': 'down', 
                'left': 'left',
                'right': 'right',
                'ok': 'enter',
                'back': 'backspace'}

mp_holistic = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

model = load_model('action.h5')

# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(40,126)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# model.load_weights('action.h5')

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

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

colors = [(245,117,16), (117,245,16), (16,117,245), (16,217,245), (116,117,245), (116,217,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


sequence = []
sentence = []
predictions = []
threshold = 0.5
prev_len_sentence = 0

cap = cv2.VideoCapture(0)
with mp_holistic.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)
        
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
                    
                    if len(sentence) != prev_len_sentence:
                        pag.press(key_actions[actions[np.argmax(res)]])
                        prev_len_sentence = len(sentence)
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
        
        cv2.imshow('Detector', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()