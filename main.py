import cv2
from imutils import face_utils
import dlib


# Vamos inicializar um detector de faces (HOG) para então
# fazer a predição dos pontos da nossa face.
p = "modelo/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

#armazena o diretorio onde a imagem de teste será armazenada
dir_image = "../controller_webcam/images/Foto_teste.png"

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
            # faça a predição e então transforme isso em um array do numpy.
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
    
            # desenhe na imagem cada cordenada(x,y) que foi encontrado.
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

 
        cv2.imshow("Controller Webcam", frame)

        # Cria um delay de 5 milisegundos
        # Armazena em key a tecla pressionada do teclado
        key = cv2.waitKey(5)
        
        # 27 -> numero da tecla ESC
        if key == 27:
            #salva um frame da camera na pasta images
            cv2.imwrite(dir_image, frame)
            break

#Finaliza a conexão com a Webcam
webcam.release()
#Fecha a janela de exibição da Webcam
cv2.destroyAllWindows()