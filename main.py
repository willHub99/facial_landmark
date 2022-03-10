from collections import OrderedDict
from imutils import face_utils
from numpy import block
from scipy.spatial import distance as dist
import cv2
import pickle
import dlib
from plot import PlotEAR

def eye_aspect_ratio(eye):
    # realiza o calculo da distancia euclidiana dos pontos verticais
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# realiza o calculo da distancia euclidiana dos pontos horizontais
	C = dist.euclidean(eye[0], eye[3])
	
	ear = (A + B) / (2.0 * C)

	return ear

# Indica proporção EAR do olho para indicar se a pessoa esta piscando
EYE_AR_THRESH = 0.3
# quantidade de frames que o olho deve estar abaixo do limite
EYE_AR_CONSEC_FRAMES = 20

# Contadores de quadros
COUNTER = 0

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#armazena o diretorio onde a imagem de teste será armazenada
dir_image = "../controller_webcam/images/Foto_teste.png"

# Vamos inicializar um detector de faces (HOG) para então
# fazer a predição dos pontos da nossa face.
p = "modelo/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

#dicionario que armazena a série temporal
data = {'ear': []}

#Permite que o opencv se conecte com a camera do pc
webcam = cv2.VideoCapture(0)

#Verifia se a conexão com a camera foi estabelecida
if webcam.isOpened():

    validacao, frame = webcam.read()
    
    while validacao:

        #Realizar a leitura dos dados da Webcam 
        # validação -> bool 
        # frame -> array de listas com os valores RGB de cada pixel do frame capturado pela camera
        # [[34,56,45], ... [45, 78, 23]]
        validacao, frame = webcam.read()
        #trasnforma de RGB -> Gray Scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectando as faces em preto e branco.
        rects = detector(gray, 0)
    
        # para cada face encontrada, encontre os pontos de interesse.
        for (i, rect) in enumerate(rects):

            """ (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255,0), 2) """
            # faça a predição e então transforme isso em um array do numpy.
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # pega as coordenadas do olho esquerdo
            left_eye = shape[lStart:lEnd]
            # pega as coordenadas do olho direito
            right_eye = shape[rStart:rEnd]
            #realiza o calculo da proporção do olho esquerdo
            leftEAR = eye_aspect_ratio(left_eye)
            #realiza o calculo da proporção do olho direito
            rightEAR = eye_aspect_ratio(right_eye)

            # eye avarage ratio -> proporção média dos olhos
            ear = (leftEAR + rightEAR) / 2

            # arredondando o valor de ear para duas cadas decimais
            ear_arredondado = round(ear, 2)

            data['ear'].append(ear_arredondado)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "ALERTA [FADIGA!!!]", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                COUNTER = 0

            for(name, (i,j)) in FACIAL_LANDMARKS_IDXS.items():
                # desenhe na imagem cada cordenada(x,y) referentes aos marcos do FACIAL_LANDMARKS_IDXS.
                for (x, y) in shape[i:j]:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
        cv2.imshow("Controller Webcam", frame)

        # Cria um delay de 5 milisegundos
        # Armazena em key a tecla pressionada do teclado
        key = cv2.waitKey(5)
        
        # 27 -> numero da tecla ESC
        if key == 27:
            save_ear = pickle.dumps(data['ear'])
            teste = PlotEAR(save_ear)
            teste.PlotEAR()
            #salva um frame da camera na pasta images
            cv2.imwrite(dir_image, frame)
            break

#Finaliza a conexão com a Webcam
webcam.release()
#Fecha a janela de exibição da Webcam
cv2.destroyAllWindows()