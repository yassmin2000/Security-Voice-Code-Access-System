import random
import math
import numba
from scipy.stats import zscore
from scipy.signal import correlate2d , stft
from sklearn import preprocessing
import os
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import  QApplication, QMainWindow, QShortcut, QFileDialog , QSplitter , QFrame , QSlider
from scipy.signal import spectrogram
from scipy.signal import resample
import sys
from PyQt5.QtGui import QIcon, QKeySequence
from mainwindow import Ui_MainWindow  
from pyqtgraph import PlotWidget, ROI

import numpy as np
import pandas as pd
from scipy.io import wavfile
import pyqtgraph as pg
from scipy.fftpack import rfft, rfftfreq, irfft , fft , fftfreq
from PyQt5.QtCore import pyqtSlot
import sounddevice as sd
import librosa


class MplCanvas(FigureCanvasQTAgg):
    
    def __init__(self, parent=None, width=5, height=1, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
    

 
class MyWindow(QMainWindow):   
    
    def __init__(self ):
        super(MyWindow , self).__init__()
      
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  
        self.spectrogram_canvas = MplCanvas(self)
        self.ui.layout_spectogrm.addWidget(self.spectrogram_canvas)
        self.ui.btn_record.clicked.connect(self.record_input_voice)
        
        self.input_fs = 22050
        self.access = False
        
        self.who_can_access = []
        
       
        self.sentenses_mfcc = {}
        

        # self.mfcc_open_middle_door = {}
        # self.mfcc_unlock_the_gate = {}
        # self.mfcc_grant_me_access = {}
        
        # self.mfcc_sara = {}
        # self.mfcc_reem = {}
        # self.mfcc_yasmeen = {}
        # self.mfcc_amir = {}
        # self.mfcc_hesham = {}
        # self.mfcc_mahmoud = {}
        # self.mfcc_ahmed = {}
        # self.mfcc_osama = {}


        # self.folder_path_open_midlle_door = "./open_middle_door"
        # self.folder_path_unlock_the_gate = "./unlock_the_gate"
        # self.folder_path_grant_me_access = "./grant_me_access"

        # self.folder_path_sara = "./sara"
        # self.folder_path_reem = "./reem"
        # self.folder_path_yasmeen = "./yasmeen"
        # self.folder_path_osama = "./osama"
        # self.folder_path_hesham = "./hesham"
        # self.folder_path_mahmoud = "./mahmoud"
        # self.folder_path_ahmed = "./ahmed"
        # self.folder_path_amir = "./amir"

        self.sentense_mfcc = []

        self.sentense_spect = []


        self.user_mfcc = []
        
        self.sentenses = ["open_middle_door" , "unlock_the_gate" ,"grant_me_access"]
        self.users = [ "sara" , "reem" , "yasmeen" , "amir" , "osama" , "hesham" , "mahmoud" , "ahmed"  ]

        for user in self.users:
            user_checkbox = getattr(self.ui, f"chkBox_{user}")
            user_checkbox.stateChanged.connect(lambda state, user=f"{user}": self.users_to_access(user, state == Qt.Checked))

        # for i,  file_name in enumerate(os.listdir(self.voice_folder_path)):
        #     file_path = os.path.join(self.voice_folder_path, file_name)
        #     audio_data , sampleRate = librosa.load(file_path)
        #     mfcc = self.extract_feature_points(audio_data , sampleRate)
        #     self.sentenses_mfcc[file_name] = mfcc
        #     print(f"file{i} :{mfcc.shape}")
            
        for i ,  sentense_folder in enumerate(self.sentenses)  :
            folder_path = f"./{sentense_folder}"
            self.sentense_spect.append({sentense_folder : {}})
            for j , file_name in enumerate(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file_name)
                audio_data , sample_rate = librosa.load(file_path)
                print(audio_data.shape)
                spect = self.calc_spect(audio_data , sample_rate)
                # dict = getattr(self , f"mfcc_{sentense_folder}")
                # dict[file_name] = spect
                self.sentense_spect[i][sentense_folder][f"{file_name.split('_')[0]}_{j}"] = spect
                # print(f"file :{file_name}")
                # print(f"file_name: {file_name}, {mfcc}")
                
            # print(self.sentense_mfcc)

        for i ,  sentense_folder in enumerate(self.sentenses)  :
            folder_path = f"./{sentense_folder}"
            self.sentense_mfcc.append({sentense_folder : {}})
            for j , file_name in enumerate(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file_name)
                audio_data , sample_rate = librosa.load(file_path)
                mfcc = self.extract_feature_points(audio_data , sample_rate)
                # dict = getattr(self , f"mfcc_{sentense_folder}")
                # dict[file_name] = mfcc
                self.sentense_mfcc[i][sentense_folder][f"{file_name.split('_')[0]}_{j}"] = mfcc
                # print(f"file :{file_name}")
                # print(f"file_name: {file_name}, {mfcc}")
                
        #     print(self.sentense_mfcc)
        
        
        
        for i , user_folder in enumerate(self.users):
            folder_path = f"./{user_folder}"
            self.user_mfcc.append({user_folder:{}})
            for j , file_name in enumerate(os.listdir(folder_path)):
                file_path = os.path.join(folder_path , file_name)
                audio_data , sample_rate = librosa.load(file_path)
                mfcc = self.extract_feature_points(audio_data , sample_rate)
                x = file_name.split('_')
                self.user_mfcc[i][user_folder][f"{file_name.split('_')[1]}_{j}"] = mfcc



        QShortcut(QKeySequence("r"), self).activated.connect(self.record_input_voice)

    
    def record_input_voice(self): # read the voice signals
        # return the audio_data
        self.ui.progressBar.setProperty("value" , 0)
        duration = 2
        input_audio = sd.rec(int(duration * self.input_fs), samplerate=self.input_fs, channels=1, dtype=np.int16)
        input_audio = np.squeeze(input_audio)

        # self.print_access_or_denied()
        sd.wait()
        self.plot_spectogram(input_audio)
        
        # get mfcc 
        mfcc = self.extract_feature_points(input_audio , self.input_fs)
        spect = self.calc_spect(input_audio , self.input_fs)

        print(f"input mfcc :{mfcc.shape}")
        user_prob = self.featurepoints_corrlation_for_users(mfcc)
        # sent_prob = {}
        sent_prob = self.featurepoints_corrlation_for_sentences_spect(spect)
        # sent_prob = self.featurepoints_corrlation_for_sentences(mfcc)
        
        max_socre_sent , sent = self.get_score_from_dict(sent_prob)
        max_socre_user , who = self.get_score_from_dict(user_prob)
        
        self.print_access_or_denied(max_socre_sent, sent ,  max_socre_user ,who )
        
        # if max_socre_sent > 0.33 and max_socre_user > 0.33:
            
            
        
        
        self.print_sentense_scores(sent_prob , user_prob)
     
        

        
        self.access = True  # here i will use the model to predict if the access is denied or not
        # self.print_access_or_denied()
        
        return input_audio
    
    
    def plot_spectogram(self , audio_data): # draw the spectogram of the input voice 
        self.spectrogram_canvas.axes.clear()
        self.spectrogram_canvas.axes.specgram(audio_data, Fs=self.input_fs)
        self.spectrogram_canvas.draw()
        
        
    def calc_spect(self , audio_data , sample_rate):     
        audio_data = audio_data.astype(np.float32)   
        stft = np.abs(librosa.stft(audio_data))
        stft_db = librosa.amplitude_to_db(np.abs(stft))
        spectrogram = librosa.feature.melspectrogram(S=stft_db, sr=sample_rate) # this is where the warn is 
        # spectrogram = librosa.feature.melspectrogram(y=audio_data.astype(np.float32), sr=sample_rate) # this is where the warn is 
        spectrogram = spectrogram.T
        scaler = preprocessing.StandardScaler()
        spectrogram_normalized = scaler.fit_transform(spectrogram)

        # Transpose the matrix back to the original shape
        spectrogram_normalized = spectrogram_normalized.T

        # spectrogram_normalized = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))

        x = np.max(spectrogram_normalized)
        y = np.min(spectrogram_normalized)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram_normalized.flatten()
        # return spectrogram_db
     

    '''mode 1 using mfcc'''
    def extract_feature_points(self , audio_data , sample_rate): # get the feature points from the spectogram
        audio_data = audio_data.astype(np.float32)
        # mfcc = np.mean(librosa.feature.mfcc(y = audio_data , sr = sample_rate , n_mfcc=50), axis=0)
        # mfcc = (librosa.feature.mfcc(y=audio_data.flatten(), sr=sample_rate, n_mfcc=50))
        mfcc = (librosa.feature.mfcc(y=audio_data.flatten(), sr=sample_rate, n_mfcc=13))

        mfcc = mfcc.T
        scaler = preprocessing.StandardScaler()
        mfcc_normalized = scaler.fit_transform(mfcc)

        # Transpose the matrix back to the original shape
        mfcc_normalized = mfcc_normalized.T

        # mfcc = mfcc.flatten()
        # print(f"")

        return mfcc_normalized.flatten()
        # pass
    
    # @numba.jit(parallel=True)
    def featurepoints_corrlation_for_sentences_spect(self , input_specto):
        max_corr_dict = {}
        avg = 0

        num = 0
        for i ,sentense in enumerate(self.sentense_spect):
            tot_corr = 0
            max_corr_for_each_sent = 0
            num += 100/3
            for user , spect in sentense[self.sentenses[i]].items():
                # print(user)
                
                corr_arr = np.correlate(input_specto, spect, mode='full')
                max_corr_position = np.unravel_index(np.argmax(corr_arr), corr_arr.shape)

                # Get the maximum correlation value itself
                corr = corr_arr[max_corr_position]

                corr_arr__ = np.correlate(spect, spect, mode='full')
                max_corr_position__ = np.unravel_index(np.argmax(corr_arr__), corr_arr__.shape)

                # Get the maximum correlation value itself
                corr__ = corr_arr__[max_corr_position__]
                # max_corr_index = np.argmax(corr_arr)
                # corr = np.max(corr_arr)
                # normalized_corr = np.max(corr_arr) / np.sqrt(np.sum(input_specto**2) * np.sum(spect**2))

                # Convert normalized correlation to a percentage
                # similarity_percentage = (normalized_corr + 1) / 2 * 100
                similarity_percentage = corr / corr__

                # corr = corr_arr[max_corr_index]
                
                # corr_arr__ = correlate2d(spect, spect, mode='full')
                # max_corr_index__ = np.argmax(corr_arr__)
                # # corr__ = corr_arr__[max_corr_index__]
                # corr__ = np.max(corr_arr__)
                # percentage_corr = corr/corr__
                
                tot_corr += similarity_percentage
                
                if similarity_percentage > max_corr_for_each_sent:
                    max_corr_for_each_sent = similarity_percentage

                print(f"{sentense.keys()} :{user} : {similarity_percentage}")
                # print(f"{sentense.keys()} :{user} : {corr__}")
                # print(f"{sentense.keys()} :{user} : {corr}")
            self.ui.progressBar.setProperty("value", num)
            max_corr_dict[f"{list(sentense.keys())[0]}"] = max_corr_for_each_sent

            # print(f":{user} : {tot_corr / num}")
            print(f"max = {max_corr_for_each_sent}")
            print("_________")
        print(max_corr_dict)
        for x , y in max_corr_dict.items():
            print(x)
            print(y)
                
       
        return max_corr_dict
        
        
    def featurepoints_corrlation_for_sentences(self , input_mfcc ): #compare between the inpus voice signal and the feature point from other spectograms
        # correlation = np.corrcoef(self.sentenses_mfcc['test.wav'], mfcc)[0, 1] 
        # print(f"correlation{correlation}")
        
        # pass        
        # correlation = np.corrcoef(self.sentenses_mfcc['sara_open_middle_door_1.wav'], mfcc)[0, 1] 
        # print(f" correlation  {correlation}")
        # correlation = np.corrcoef(self.sentenses_mfcc['sara_grant_me_access_1.wav'], mfcc)[0, 1] 
        # print(f" correlation  {correlation}")
        # correlation = np.corrcoef(self.sentenses_mfcc['sara_unlock_the_gate_1.wav'], mfcc)[0, 1] 
        # print(f" correlation  {correlation}") # till here not cross corrlation

        # max = 0
        # for sentence_name, sentence_mfcc in self.sentenses_mfcc.items():
        #     correlation = np.correlate(sentence_mfcc, mfcc, mode='full')
        #     correlation_ = np.correlate(sentence_mfcc, self.sentenses_mfcc[sentence_name], mode='full')
        #     max_corr_index = np.argmax(correlation)
        #     max_corr_index_ = np.argmax(correlation_)
        #     # normalized_correlation = zscore(correlation)

        #     # The similarity score is the maximum value of the cross-correlation
        #     similarity_score = correlation_[max_corr_index_] 
        #     similarity_score = correlation[max_corr_index] / correlation_[max_corr_index_]
        #     probability_score = 1 / (1 + np.exp(-correlation))
        #     if similarity_score > max:
        #         max = similarity_score
        #         which = sentence_name
        #     # print(f"Probability score for {sentence_name}: {probability_score}")
        #     print(f"cross corrlation  {sentence_name}: {similarity_score}")
        # print(f"max   {max} , who {which}")
        # print("_________")
        max_corr_dict = {}
        avg = 0
        tot_corr = 0
        num = 0
        for i ,sentense in enumerate(self.sentense_mfcc):
            max_corr_for_each_sent = 0
            for user , mfcc in sentense[self.sentenses[i]].items():
                num += 1
                # print(user)
                corr_arr = np.correlate(input_mfcc, mfcc, mode='full')
                max_corr_index = np.argmax(corr_arr)
                corr = corr_arr[max_corr_index]
                
                corr_arr__ = np.correlate(mfcc, mfcc, mode='full')
                max_corr_index__ = np.argmax(corr_arr__)
                corr__ = corr_arr__[max_corr_index__]

                percentage_corr = corr/corr__
                
                tot_corr += percentage_corr
                
                if percentage_corr > max_corr_for_each_sent:
                    max_corr_for_each_sent = percentage_corr

                print(f"{sentense.keys()} :{user} : {percentage_corr}")
                # print(f"{sentense.keys()} :{user} : {corr__}")
                # print(f"{sentense.keys()} :{user} : {corr}")
            max_corr_dict[f"{list(sentense.keys())[0]}"] = max_corr_for_each_sent

            # print(f":{user} : {tot_corr / num}")
            print(f"max = {max_corr_for_each_sent}")
            print("_________")
        print(max_corr_dict)
        for x , y in max_corr_dict.items():
            print(x)
            print(y)
                
       
        return max_corr_dict
    
    def featurepoints_corrlation_for_users(self , input_mfcc):
        max_corr_dict = {}
        avg = 0
        tot_corr = 0
        num = 0
        for i ,user in enumerate(self.user_mfcc):
            max_corr_for_each_user = 0
            for sentense , mfcc in user[self.users[i]].items():
                num += 1
                # print(user)
                corr_arr = np.correlate(input_mfcc, mfcc, mode='full')
                max_corr_index = np.argmax(corr_arr)
                # corr = np.max(corr_arr)
                corr = corr_arr[max_corr_index]
                
                corr_arr__ = np.correlate(mfcc, mfcc, mode='full')
                max_corr_index__ = np.argmax(corr_arr__)
                corr__ = corr_arr__[max_corr_index__]
                # corr__ = np.max(corr_arr__)
                percentage_corr = corr/corr__
                
                tot_corr += percentage_corr
                
                if percentage_corr > max_corr_for_each_user:
                    max_corr_for_each_user = percentage_corr

                print(f"{user.keys()} :{sentense} : {percentage_corr}")
                # print(f"{user.keys()} :{user} : {corr__}")
                # print(f"{user.keys()} :{user} : {corr}")
            max_corr_dict[f"{list(user.keys())[0]}"] = max_corr_for_each_user

            # print(f":{user} : {tot_corr / num}")
            print(f"max = {max_corr_for_each_user}")
            print("_________")
        print(max_corr_dict)
        for x , y in max_corr_dict.items():
            print(x)
            print(y)
                
       
        return max_corr_dict
    
        pass

    def print_sentense_scores(self , probs_sent , prop_user): # see how close the input signal and to the 3 sentences or the 8 user voices
        probs_sent = self.print(probs_sent)
        for sentsnce , prob in probs_sent.items():
            cell = getattr(self.ui , f"lbl_prop_{sentsnce }")
            # cell.setStyleSheet("")
            # if prob > .7:
            #     cell.setStyleSheet("color: green")
            #     cell.setText(str(round(prob * 100, 2)))
            #     continue
                
            cell.setText(str(round(prob , 2)))
            
        all_probabilities_below_threshold = True  
        
        for user , prob in prop_user.items():
            
            cell = getattr(self.ui , f"lbl_prob_{user }")
            # cell.setStyleSheet("")
            # if prob > .33:
            #     cell.setStyleSheet("color: green")
            #     cell.setText(str(round(prob * 100, 2)))
            #     continue
                
            cell.setText(str(round(prob , 2)))
        for user , prob in prop_user.items():
            if prob >= 0.33:
                all_probabilities_below_threshold = False
                break 
            
        if all_probabilities_below_threshold:
            self.ui.other_lbl.setText("other")
        else :
            self.ui.other_lbl.setText("")
                

        
        # pass
    
    
    def users_to_access(self , user , ischecked): # update how can access from the users 
        print(ischecked)
        if ischecked:
            self.who_can_access.append(user)
        else:
            self.who_can_access.remove(user)
            
        print(self.who_can_access)


    def print_access_or_denied(self , max_sent ,sent, max_user ,who): #print if the user is allowed to access or not 
        if max_sent > 0.70 and max_user > 0.33  and who in self.who_can_access:
            self.ui.lbl_access.setText(f'<font color="green">wlecome {who}</font>')
            

        else:
            self.ui.lbl_access.setText('<font color="red">Access Denied</font>')
                
    def get_score_from_dict(self , dict):
        max = 0
        for _ , score in dict.items():
            if score > max :
                max = score
                user_sent = _
        return max , user_sent

    def print(self , dict):
        max_key, max_value = max(dict.items(), key=lambda item: item[1])
        for x , y in dict.items():
            dict[x] = math.floor(y*10)/10 

        dict[max_key] = math.ceil(max_value*10)/10 
        
        return dict
            
        
            
    

def main():
    app = QApplication(sys.argv)
    window = MyWindow() 
   
   
    window.showMaximized()
    window.show()
    
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()