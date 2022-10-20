import cv2
import numpy as np
from time import sleep
#from constantes import *

# ---------- Constantes ----------

largo_min = 80  # Lorgo minimo del rectangulo
altura_min = 80  # Altura minima do retangulo
offset = 6  # Error permitido entre pixel
posicion_linea = 550  # Posicion de la linea de conteo
delay = 60  # FPS do vídeo
detec = []

# ---------------------------------
def pega_centro(x, y, largo, altura):
    x1 = largo // 2
    y1 = altura // 2
    cx = x + x1
    cy = y + y1
    return cx, cy

def set_info(detec):
    global vehiculos
    for (x, y) in detec:
        if (posicion_linea + offset) > y > (posicion_linea - offset):
            vehiculos += 1
            cv2.line(frame1, (25, posicion_linea), (1200, posicion_linea), (0, 127, 255), 3)
            detec.remove((x, y))

def show_info(frame1, dilatada):
    text = f'Vehiculos: {vehiculos}'
    cv2.putText(frame1, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame1)
    #cv2.imshow("Detectar", dilatada)  # Para ver la deteccion con los filtros aplicados

vehiculos = camiones = 0
cap = cv2.VideoCapture('autosPasando.mp4')
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()  # Toma el fondo y resta de lo que se mueve

while True:
    ret, frame1 = cap.read()  # Pega cada frame de vídeo
    tempo = float(1 / delay)
    sleep(tempo)  # Da un retraso entre cada procesamiento.
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Toma el frame y lo conviernte en escala de grises
    blur = cv2.GaussianBlur(grey, (3, 3), 5)  # Hace un desenfoque para intentar quitar las imperfecciones de la imagen
    img_sub = subtracao.apply(blur)  # Resta la imagen aplicada en el desenfoque.
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))  # "Engrosa" lo que queda de la resta
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  #Crea una matriz de 5x5, donde el formato de matriz entre 0 y 1 forma una elipse en el interior
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)  # Intenta llenar todos los "agujeros" en la imagen.
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    contorno, img = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (25, posicion_linea), (1200, posicion_linea), (255, 127, 0), 3)
    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= largo_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centro = pega_centro(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

    set_info(detec)
    show_info(frame1, dilatada)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()