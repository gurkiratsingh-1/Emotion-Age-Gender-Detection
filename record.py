
import tkinter as tk
from tkinter import Button,Label
import pyaudio
import wave
import pandas as pd
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_json
import matplotlib.pyplot as plt
import librosa.display
from PIL import Image, ImageTk

json_file = open('C:/Users/Gurkirat/Speech-Emotion-Analyzer-master/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:/Users/Gurkirat/Speech-Emotion-Analyzer-master/saved_models/Emotion_Voice_Detection_Model.h5")
print("Loaded model from disk")
actualvalues={
'0':'Female Angry',
'1':'Female Calm',
'2':'Female Fearful',
'3':'Female Happy',
'4':'Female Sad',
'5':'Male Angry',
'6':'Male Calm',
'7':'Male Fearful',
'8':'Male Happy',
'9':'Male Sad'
}

mic="C:/Users/Gurkirat/Desktop/mic.jpg"
pre="C:/Users/Gurkirat/Desktop/predict.jpg"
        
class ui:
    def __init__(self):

        
        self.root = tk.Tk()
        self.root.title('Recorder')
        
        self.root.geometry('550x370')
        '''
        self.img5 = ImageTk.PhotoImage(Image.open(mic))
        self.panel = Label(self.root, image = self.img5,bg="gray25")
        self.panel.image = self.img5
        self.panel.place(x=10,y=10)
        
        self.img6 = ImageTk.PhotoImage(Image.open(pre))
        self.panel = Label(self.root, image = self.img6,bg="gray25")
        self.panel.image = self.img6
        self.panel.place(x=270,y=10)
        '''
        self.button = tk.Button(self.root,padx=16,pady=4,bd=4,fg="red",font=('arial',18,'bold'),width=14,height=3,bg="gray25", text="Recording",command=self.record)
        self.button.place(x=10,y=220)


        self.slogan = tk.Button(self.root,padx=16,pady=4,bd=4,fg="white",font=('arial',18,'bold'),width=14,height=3,bg="gray25",text="Predict",command=self.test)

        self.slogan.place(x=270,y=220)
        

        
        #self.w = tk.Label(self.root, textvariable=self.b,fg="blue",font=('comic sans ms',22,'bold'),width=24,height=3)
        #self.w.place()

        
        

        

        self.root.mainloop()

    def record(self):

        CHUNK = 1024 
        FORMAT = pyaudio.paInt16 #paInt8
        CHANNELS = 2 
        RATE = 44100 #sample rate
        RECORD_SECONDS = 4
        WAVE_OUTPUT_FILENAME = "output10.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK) #buffer
    
        print("* recording")
    
        frames = []
    
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data) # 2 bytes(16 bits) per channel
    
        print("* done recording")
    
        stream.stop_stream()
        stream.close()
        p.terminate()
    
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
    def test(self):
    
        X, sample_rate = librosa.load('output10.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
        featurelive = mfccs
        livedf2 = featurelive
        livedf2= pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame().T
        twodim= np.expand_dims(livedf2, axis=2)
        livepreds = loaded_model.predict(twodim, 
                                 batch_size=32, 
                                 verbose=1)

        livepreds1=livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()

        lb = LabelEncoder()
        liveabc=lb.fit_transform(liveabc)
        livepredictions = (lb.inverse_transform((liveabc)))
        livepredictions=str(list(livepredictions))[1:-1]
        ans=livepredictions
        for key,value in actualvalues.items():
            if(ans==key):
                break
        root1 = tk.Tk()
        root1.title('Predicted Result')
        
        root1.geometry('460x200')
        
        w = tk.Label(root1, text=value,fg="blue",font=('comic sans ms',22,'bold'),width=24,height=3)
        w.place(x=10,y=10)
        
        




