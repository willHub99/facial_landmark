from collections import OrderedDict
from imutils import face_utils
from numpy import block
from scipy.spatial import distance as dist
import cv2
import pickle
import dlib
from plot import Plot
from collections import OrderedDict
from imutils import face_utils
import os
import datetime
from threading import Thread
import playsound
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#=====================================================================================
#===================classe responsavel por controlar acoes da camera==================
#=====================================================================================


class WebcamController:

    #==========================================================
    #===================ATRIBUTOS DA CLASSE====================
    #==========================================================

    """ FACIAL_LANDMARKS_IDXS = OrderedDict([
	    ("mouth", (48, 68)),
	    ("right_eyebrow", (17, 22)),
	    ("left_eyebrow", (22, 27)),
	    ("right_eye", (36, 42)),
	    ("left_eye", (42, 48)),
	    ("nose", (27, 35)),
	    ("jaw", (0, 17))
    ]) """
    
    FACIAL_LANDMARKS_IDXS = OrderedDict([
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
    ])

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Vamos inicializar um detector de faces (HOG) para então
    # fazer a predição dos pontos da nossa face.
    model = "modelo/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model)

    #Permite que o opencv se conecte com a camera do pc
    webcam = cv2.VideoCapture(0)

    #armazena a data atual
    now = datetime.datetime.now()

    #formata data para ano-mes-dia/hora-minuto-segundo
    date_format = str(now.year) + '-' + str(now.month) + '-' + str(now.day) + '_' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second)

    #armazena o diretório base do projeto
    dir_base = os.getcwd()

    #armazena o diretorio onde a imagem colorida será armazenada
    dir_image_colorful = f"{dir_base}/assets/images/colorful/{date_format}.png"

    #armazena o diretorio onde a imagem colorida será armazenada
    dir_image_gray_scale = f"{dir_base}/assets/images/gray/{date_format}.png"

    #armazena o diretorio onde o arquivo dos dados do dicionario data sera armazenado
    dir_ear_file = f"{dir_base}/assets/files/ear/ear_{date_format}.pkl"

    #armazena o diretorio onde o arquivo dos dados do dicionario time sera armazenado
    dir_time_file = f"{dir_base}/assets/files/time/time_{date_format}.pkl"

    #armazena o diretorio onde o arquivo dos dados do dicionario square sera armazenado
    dir_square_file = f"{dir_base}/assets/files/square/square_{date_format}.pkl"

    # Limiar utilizado para delimitar evento de sonolência
    EYE_EAR_THRESH = 0.3
    # quantidade de frames que o olho deve estar abaixo do limite
    EYE_EAR_CONSEC_FRAMES = 30

    # Contadores de quadros
    COUNTER = 0

    #variavel que armazena o valor atual da onda quadrada (0,5 -> olho aberto, 0 -> olho fechado)
    switch_square = 1

    #armazena os valores de switch_square
    square = {'square': []}

    #dicionario que armazena a série temporal dos valores EAR
    data = {'ear': []}

    time = {'time': []}

    ALARM = "assets/sound-alarm/alarm.wav"

    ALARM_ON = False

    #==========================================================
    #=====================METODOS DA CLASSE====================
    #==========================================================

    def sound_alarm(self):
        alarm = path=self.ALARM
        # play an alarm sound
        playsound.playsound(alarm)

    def eye_aspect_ratio(self, eye):
        # realiza o calculo da distancia euclidiana dos pontos verticais
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # realiza o calculo da distancia euclidiana dos pontos horizontais
        C = dist.euclidean(eye[0], eye[3])
        
        return (A + B) / (2.0 * C)

    def calculateAverageAndInsertIntoTheEARList(self, leftEAR: float, rightEAR: float):
        # eye avarage ratio -> proporção média dos olhos
        ear = (leftEAR + rightEAR) / 2

        # arredondando o valor de ear para duas cadas decimais
        #ear_arredondado = round(ear, 2)
        ear_arredondado = ear

        '''if self.data['ear'] and (max(self.data['ear']) - min(self.data['ear'])):
            std = ( ear_arredondado - min(self.data['ear']) )/ (max(self.data['ear']) - min(self.data['ear']))
            ear_scaled = std * (1 - 0) + 0
        else:
            ear_scaled = ear_arredondado'''

        # adiciona o valor de EAE arredondado na serie temporal
        self.data['ear'].append(ear)
    
        return ear_arredondado

    def updateEyeEarThresh(self):

        dados = {
            "ear": self.data['ear']
        }

        #cria um pandas dataframe
        dados = pd.DataFrame(dados)

        #define um novo valor para ear (média - variância)
        new_ear = round(dados['ear'].mean(), 4) - round(dados['ear'].var(), 4)

        #instância o modelo de floresta de isolação
        isolation_model = IsolationForest(contamination = 0.03)
        #treina o modelo 
        isolation_model.fit(dados)
        #rotula dados:
        #  1: dados normais
        #  -1: valoresa discrepantes 
        predictions = isolation_model.predict(dados)

        if -1 in list(predictions):
            outlier = len(list(predictions)) - 1 - list(predictions)[::-1].index(-1)
            return dados['ear'][outlier]

        return round(new_ear, 2)

    def setTime(self):
        # armazena data atual 
        now = datetime.datetime.now()
        self.time["time"].append(now)


    # salva a figura colorida
    def savePictures(self, frameColoforful, frameGrayScale):

        #salva um frame da camera colorido na pasta colorful
        cv2.imwrite(self.dir_image_colorful, frameColoforful)
        #salva um frame da camera colorido na pasta gray_scale
        cv2.imwrite(self.dir_image_gray_scale, frameGrayScale)

    def savePicleFiles(self):
        with open(self.dir_ear_file, 'wb') as handle: pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.dir_time_file, 'wb') as handle: pickle.dump(self.time, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.dir_square_file, 'wb') as handle: pickle.dump(self.square, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def initThreadAlarmSound(self):
        # ligar alarme
        if not self.ALARM_ON:
            self.ALARM_ON = True
            t = Thread(target=self.sound_alarm)
            t.start()

    #==========================================================
    #=================MANIPULACAO DA CAMERA====================
    #==========================================================
    def cameraReadLoop(self):
        #Verifia se a conexão com a camera foi estabelecida
        if self.webcam.isOpened():

            validacao, frame = self.webcam.read()
            
            while validacao:

                #Realizar a leitura dos dados da Webcam 
                # validação -> bool 
                # frame -> array de listas com os valores RGB de cada pixel do frame capturado pela camera
                # [[34,56,45], ... [45, 78, 23]]
                validacao, frame = self.webcam.read()

                #trasnforma de RGB -> Gray Scale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detectando as faces em preto e branco.
                rects = self.detector(gray, 0)

                #armazena no dicionario time os valores de tempo da serie temporal
                self.setTime()
            
                # para cada face encontrada, encontre os pontos de interesse.
                for (i, rect) in enumerate(rects):

                    """ (x, y, w, h) = face_utils.rect_to_bb(rect)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255,0), 2) """
                    # faça a predição e então transforme isso em um array do numpy.
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # pega as coordenadas do olho esquerdo
                    left_eye = shape[self.lStart:self.lEnd]
                    # pega as coordenadas do olho direito
                    right_eye = shape[self.rStart:self.rEnd]
                    #realiza o calculo da proporção do olho esquerdo
                    leftEAR = self.eye_aspect_ratio(left_eye)
                    #realiza o calculo da proporção do olho direito
                    rightEAR = self.eye_aspect_ratio(right_eye)

                    ear = self.calculateAverageAndInsertIntoTheEARList(leftEAR, rightEAR)

                    #atualiza o valor do EYE_EAR_THRESH
                    EYE_EAR_THRESH = self.updateEyeEarThresh()

                    #armazena os valores da onda quadrada no dicionario esquare
                    self.square["square"].append(self.switch_square)

                    if ear < EYE_EAR_THRESH:
                        self.switch_square = 0
                        COUNTER += 1
                        if COUNTER >= self.EYE_EAR_CONSEC_FRAMES:
                            self.savePictures(frame, gray)
                            #aciona a sirene de alerta
                            self.initThreadAlarmSound()

                            cv2.putText(frame, "!!! ALERTA FADIGA !!!" , (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                    else:

                        cv2.putText(frame, "!!! Status: Atento !!!" , (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                        self.switch_square = 1
                        COUNTER = 0
                        self.ALARM_ON = False

                    for(name, (i,j)) in self.FACIAL_LANDMARKS_IDXS.items():
                        # desenhe na imagem cada cordenada(x,y) referentes aos marcos do FACIAL_LANDMARKS_IDXS.
                        for (x, y) in shape[i:j]:
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
                    cv2.putText(frame, "EYE_EAR_THRESH: {:.2f}".format(EYE_EAR_THRESH), (365, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
                cv2.imshow("Controller Webcam", frame)

                # Cria um delay de 5 milisegundos
                # Armazena em key a tecla pressionada do teclado
                key = cv2.waitKey(5)
                
                # 27 -> numero da tecla ESC
                if key == 27:

                    self.savePicleFiles()
                    teste = Plot()
                    teste.Plot(self.date_format)

                    break

        #Finaliza a conexão com a Webcam
        self.webcam.release()
        #Fecha a janela de exibição da Webcam
        cv2.destroyAllWindows()


def main():
    webcamController = WebcamController()
    webcamController.cameraReadLoop()

if __name__ == "__main__": main()
