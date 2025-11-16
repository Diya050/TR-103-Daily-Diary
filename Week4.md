# **Day - 22**

### **1. Definition**

NLP is a field of Artificial Intelligence that enables computers to understand, interpret, and generate human language.

### **2. Components of NLP**

* **Syntax** – Structure and grammar of language
* **Semantics** – Meaning of words and sentences
* **Pragmatics** – Context and intent behind language

### **3. Common NLP Tasks**

* **Tokenization** – Splitting text into words or sentences
  Example:
  Text: "I love machine learning" → ["I", "love", "machine", "learning"]

* **Stopword Removal** – Removing common words like "the", "is", "and"
  Example:
  "The book is interesting" → ["book", "interesting"]

* **Stemming** – Reducing words to their root form
  Example: running, runs → run

* **Lemmatization** – Converting words to meaningful root forms
  Example: better → good

### **4. Applications**

* Chatbots
* Translation systems
* Text classification
* Spam detection



# **Day - 23**

### **1. Bag of Words (BoW)**

Represents text as the frequency of each word, without considering order.

**Example:**
Text: "Machine learning is fun"
Vector: {"machine": 1, "learning": 1, "fun": 1}

### **2. TF-IDF (Term Frequency–Inverse Document Frequency)**

Gives importance to rare but meaningful words.

**Example:**
If "machine" is rare in all documents, it gets a high score.

### **3. Word Embeddings**

Words are converted into dense numeric vectors that capture meaning.

* **Word2Vec**
* **GloVe**

**Example:**
Similarity("king", "queen") > Similarity("king", "apple")

### **4. Text Classification**

Predicts a label for a given text.

**Examples:**

* Spam or not spam
* Positive or negative sentiment

### **5. Named Entity Recognition (NER)**

Identifies entities like names, locations, dates.

**Example:**
Text: "Diya lives in Ludhiana"
NER Output: Diya → Person, Ludhiana → Location



# **Day - 24**

### **1. Recurrent Neural Networks (RNN)**

Neural networks designed for sequence data.

**Example Use:**
Predicting the next word in a sentence.

### **2. LSTM (Long Short-Term Memory)**

Improved RNN that handles long-term dependencies.

**Example:**
Used in language translation and speech-to-text systems.

### **3. Transformers**

A deep learning architecture that uses attention mechanisms to process text in parallel.

**Example:**
Models like BERT, GPT, T5

### **4. Attention Mechanism**

Helps model focus on important words in a sequence.

**Example:**
In the sentence "The car which I bought yesterday is fast",
attention focuses on "car" and "fast".

### **5. Modern NLP Applications**

* Text summarization
* Machine translation
* Question answering systems
* Voice assistants
* ChatGPT-style generative models

### **6. Example: Sentiment Analysis using BERT**

Input: "The product is excellent"
Output: Positive sentiment

Below is a clear, step-by-step **tutorial from Day 25 to Day 28** that will help you **develop the Self-Driving Car Deep Learning Project from scratch**.
This covers coding steps, explanations, dataset creation, preprocessing, model building, training, and testing — *exactly matching your project workflow*.



# **Day - 25**

### **Goal:** Understand the architecture and create the base project structure for developing self-driving car simulation.



## **1. Project Folder Structure**

Create a folder:

```
self-driving-car/
│── data/
│   ├── IMG/
│   └── driving_log.csv
│── model/
│── drive.py
│── train.py
│── utils.py
│── model.json
│── model.h5
```



## **2. Install Required Libraries**

```bash
pip install numpy pandas tensorflow keras flask socketio python-socketio eventlet opencv-python albumentations matplotlib
```



## **3. Understand the CNN Architecture (NVIDIA Model)**

A CNN takes raw images and predicts steering angles.

### **Key Layers**

* **Convolution Layers:** detect edges/lane lines
* **Pooling/Cropping:** remove unnecessary image areas
* **Fully Connected Layers:** convert image features into steering angle predictions



## **4. Build the CNN Model (model.py)**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5,5), strides=(2,2), activation='elu', input_shape=(66,200,3)))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='elu'))
    model.add(Conv2D(64, (3,3), activation='elu'))
    model.add(Conv2D(64, (3,3), activation='elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
```



# **Day - 26**



## **1. Collect Simulator Data**

Use **Udacity Self-Driving Car Simulator → Training Mode**
Record your driving.
This will generate:

* **IMG/*.jpg** → images
* **driving_log.csv** → steering, throttle, brake, speed



## **2. Load and Inspect Data**

```python
import pandas as pd

cols = ['center','left','right','steering','throttle','brake','speed']
data = pd.read_csv('data/driving_log.csv', names=cols)
print(data.head())
```



## **3. Preprocessing Functions (utils.py)**

### **Cropping, Resizing, YUV conversion, Normalization**

```python
import cv2
import numpy as np

def preprocess(img):
    img = img[60:135, :, :]     # crop sky & hood
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66))
    img = img / 255.0
    return img
```



## **4. Data Augmentation**

To simulate real-world conditions:

```python
import albumentations as A

augment = A.Compose([
    A.RandomBrightness(limit=0.3, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5)
])
```



## **5. Steering Angle Correction for Left & Right Cameras**

```python
correction = 0.25
```



# **Day - 27**



## **1. Image Generator (Memory Efficient)**

Sends batches of data to the model during training.

```python
def batch_generator(image_paths, steering, batch_size):
    while True:
        x = []
        y = []
        for i in range(batch_size):
            index = np.random.randint(0, len(image_paths))
            img = cv2.imread(image_paths[index])
            img = preprocess(img)
            x.append(img)
            y.append(steering[index])
        yield (np.array(x), np.array(y))
```



## **2. Split Dataset**

```python
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(data, test_size=0.2)
```



## **3. Train the Model (train.py)**

```python
from model import nvidia_model
model = nvidia_model()

history = model.fit(
    train_gen,
    steps_per_epoch=300,
    epochs=10,
    validation_data=val_gen,
    validation_steps=200
)
```



## **4. Save Model**

```python
model.save('model.h5')
```



# **Day - 28**



## **1. The Driving Server (drive.py)**

Receives images from the simulator → preprocess → predict steering → send back.

```python
from flask import Flask
from flask_socketio import SocketIO
from model import nvidia_model

app = Flask(__name__)
socketio = SocketIO(app)

model = load_model('model.h5')

@socketio.on('telemetry')
def telemetry(data):
    img = data['image']
    image = Image.open(BytesIO(base64.b64decode(img)))
    image = np.asarray(image)
    image = preprocess(image)

    steering = float(model.predict(image[None, :, :, :]))
    
    speed = float(data['speed'])
    throttle = 1.0 - (speed / 20)

    send_control(steering, throttle)
```



## **2. Send Control to Simulator**

```python
def send_control(steering, throttle):
    socketio.emit('steer', data={
        'steering_angle': str(steering),
        'throttle': str(throttle)
    })
```



## **3. Run the Server**

```bash
python drive.py
```

Then open the simulator → **Autonomous Mode**.

