#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os
import matplotlib.pyplot as plt


# In[10]:


actions = np.array([ 'up', 'down', 'left', 'right', 'ok', 'back'])

no_sequences = 30

sequence_length = 40


# In[11]:


label_map = {label:num for num, label in enumerate(actions)}


# In[12]:


parts = [
    [0, 1, 0, 5, 0, 17, 'green'],
    [5,9,9,13,13,17,'red'],
    [1,2,2,3,3,4,'blue'],
    [5,6,6,7,7,8,'yellow'],
    [9,10,10,11,11,12,'orange'],
    [13,14,14,15,15,16,'pink'],
    [17,18,18,19,19,20,'purple'],
    
    #[0+21, 1+21, 0+21, 5+21, 0+21, 17+21, 'green'],
    #[5+21,9+21,9+21,13+21,13+21,17+21,'red'],
    #[1+21,2+21,2+21,3+21,3+21,4+21,'blue'],
    #[5+21,6+21,6+21,7+21,7+21,8+21,'yellow'],
    #[9+21,10+21,10+21,11+21,11+21,12+21,'orange'],
    #[13+21,14+21,14+21,15+21,15+21,16+21,'pink'],
    #[17+21,18+21,18+21,19+21,19+21,20+21,'purple']
]

def draw_parts(ax, dx, dy, dz, parts):
    ax.plot3D([dx[parts[0]],dx[parts[1]]], [dy[parts[0]],dy[parts[1]]], [dz[parts[0]],dz[parts[1]]], parts[6])
    ax.plot3D([dx[parts[2]],dx[parts[3]]], [dy[parts[2]],dy[parts[3]]], [dz[parts[2]],dz[parts[3]]], parts[6])
    ax.plot3D([dx[parts[4]],dx[parts[5]]], [dy[parts[4]],dy[parts[5]]], [dz[parts[4]],dz[parts[5]]], parts[6])


# In[13]:


def get_coords(X):   
    dx = X[:63:3]
    dy = X[1:63:3]
    dz = X[2:63:3]
    return dx,dy,dz


# In[14]:


label_map


# In[15]:


# sequences, labels = [], []
# for action in actions:
#     for sequence in range(no_sequences):
#         window = []
#         for frame_num in range(sequence_length):
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])


# In[4]:


sequences = np.load('data/sequences.npy').tolist()
labels = np.load('data/labels.npy').tolist()


# In[24]:


np.array(labels).shape
np.array(sequences).shape


# In[25]:


np.save('sequences.npy', np.array(sequences))
np.save('labels.npy', np.array(labels))


# In[26]:


print(sequences[4])


# In[9]:


X = np.array(sequences)


# In[10]:


X.shape


# In[11]:


def only_right_hand(X):
    res = []
    for i in X:
        res_j = []
        for j in i:
            res_j.append(j[:63])
        res.append(res_j)
    return res


# In[13]:


X.shape


# In[14]:


def interp_coords(x):
    coords = []
    for num,i in enumerate(x):
        if np.count_nonzero(i) != 0:
            coords.append([i,num])
    result = []
    for i in range(63):
        result.append(np.interp(range(40),[e[1] for e in coords],[e[0][i] for e in coords]))
    return np.array(result).transpose()


# In[15]:


def draw_hand(x, parts, save=False, fixed_axes=False, dinamic_view=False):
    for num,i in enumerate(x):
        dx,dy,dz = get_coords(i)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', label=num)
        ax.scatter(dx, dy, dz)

        for i in parts:
            draw_parts(ax, dx, dy, dz, i)

        ax.view_init(260, -120)
        figure = plt.gcf()
        
        if fixed_axes:
            plt.xlim([0,1])
            plt.ylim([0,1])
            
        if dinamic_view:
            ax.view_init(260+2*num, -120+4*num)
            
        plt.show()
        if save:
            figure.savefig(os.path.join("hand_frames", "{}.jpg".format(num)))
        plt.close(fig)


# In[18]:


sample = only_right_hand(X)
draw_hand(interp_coords(sample[139]), parts, save=False, dinamic_view=False, fixed_axes=True)
#draw_hand(sample[139], parts, save=True)


# In[ ]:




