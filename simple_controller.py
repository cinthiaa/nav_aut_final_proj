"""camera_pid controller."""

#Importar las librerías que vamos a necesitar 
from controller import Display, Keyboard, Robot, Camera 
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import csv
import shutil
import time

#Captura la imagen desde la cámara de Webots 
def get_image(camera): #Definimos la función 
    raw_image = camera.getImage()  #Imagen sin procesar
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4) # Se convierte a array (alto, ancho, RGBA)
    )
    return image[:, :, :3] #Se descarta el canar alfa y solo se utiliza el RGB

#Procesamiento de imagen para detectar la línea que va a ser el automóvil
def process_image(image): #Definimos la función 
    perc_h = 0.85 # Porcentaje de altura de la imagen a considerar
    h, w, _ = image.shape # Altura y ancho de la imagen

    # Convertir a HSV para facilitar la detección de colores
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
    # Para blanco (alto brillo, baja saturación)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white) #cambio para detectar linea blanca


    #Aplicamos la máscara sobre la imagen original 
    white_only = cv2.bitwise_and(image, image, mask=mask)

    #Hacemos preprocesamiento para detectar bordes y se detecte de mejor forma la línea
    gray = cv2.cvtColor(white_only, cv2.COLOR_BGR2GRAY) #Convertimos a escala de grises
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) #Suavizamos la imagen para reducir el ruido
    edges = cv2.Canny(blurred, 30, 100) #Detección de bordes 
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=3) #Realzamos los bordes

    #Definimos la región de interés para limitar la detección de líneas
    roi = np.zeros_like(dilated)
    roi[int(perc_h * h):, :int(w/2)] = 255  
    masked_edges = cv2.bitwise_and(dilated, roi)

    #Detección de líneas con la transformada de Hough
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=25,
                            minLineLength=20, maxLineGap=20)

    #Prepararamos imagen para mostrar en el display de Webots
    
    #Convertimos a escala de grises para mostrar en el display
    gray_img_to_display = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Se define la ROI para mostrar en el display
    img_roi = np.zeros_like(gray_img_to_display)
    #Los vertices de la ROI abarcan el 60% de la altura de la imagen
    vertices = np.array([[(0,h),(0, perc_h*h), (w/2, perc_h*h), (w/2,h)]], dtype=np.int32) 
    cv2.polylines(img_roi, vertices, isClosed=True, color=105, thickness=2)
    alpha = 1
    beta = 1
    gamma = 0
    #Aplicamos la ROI a la imagen en escala de grises
    display_img = cv2.addWeighted(gray_img_to_display, alpha, img_roi, beta, gamma)


    #Inicializamos la línea más relevante (la central)
    best_line = None
    steering_needed = False
    found = False  # Variable para indicar si se encontró una línea
    lowest_y = -1
    cx = w // 2
    """
    #Seleccionamos la mejor línea (más cercana a la parte baja)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx == 0:
                dx = 1
            slope = dy / dx
            #if slope < 0.9:  # Permitir líneas suficientemente inclinadas
            print(f'{"slope:", slope, "x1:", x1, "y1:", y1, "x2:", x2, "y2:", y2}')
            y_avg = (y1 + y2) // 2
            if y_avg > lowest_y: #Elegir la más baja
                lowest_y = y_avg
                best_line = (x1, y1, x2, y2)
                cx = (x1 + x2) // 2 #Centro horizontal de la línea"""
    
    # Inicializamos listas para acumular líneas válidas
    x1_list = []
    y1_list = []
    x2_list = []
    y2_list = []
    slope_list = []
    all_list = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue  # evitamos división por cero
            slope = dy / dx
            all_list.append((slope, x1, y1, x2, y2))  # Guardamos pendiente y coordenadas
            #print(f'{"slope:", slope, "x1:", x1, "y1:", y1, "x2:", x2, "y2:", y2}')
        
        # ----- FILTRAR OUTLIERS DE SLOPE -----
        # Extraer solo slopes
        all_slopes = np.array([s for s, _, _, _, _ in all_list])
        if len(all_slopes) > 0:
            mean_slope = np.mean(all_slopes)
            std_slope = np.std(all_slopes)
            lower_bound = mean_slope - 1.5 * std_slope
            upper_bound = mean_slope + 1.5 * std_slope

            # Filtrar líneas sin outliers
        for slope, x1, y1, x2, y2 in all_list:
            if lower_bound <= slope <= upper_bound:
                #print(f'{"Sin outliers slope:", slope, "x1:", x1, "y1:", y1, "x2:", x2, "y2:", y2}')
                x1_list.append(x1)
                y1_list.append(y1)
                x2_list.append(x2)
                y2_list.append(y2)
                slope_list.append(slope)

        # ----- CONCERTIR A numpy ARRAY -----
        x1_array = np.array(x1_list)
        y1_array = np.array(y1_list)
        x2_array = np.array(x2_list)
        y2_array = np.array(y2_list) 
        slope_array = np.array(slope_list) 

        #----- si el promedio de las slopes es suficiente, calcular best lines y cx -----
        if len(slope_array) > 0:    
            mean_slope = np.mean(slope_array)

            # Calcular la línea promedio
            x1_avg = int(np.mean(x1_array))
            y1_avg = int(np.mean(y1_array))
            x2_avg = int(np.mean(x2_array))
            y2_avg = int(np.mean(y2_array))
                
            # Actualizar la mejor línea
            best_line = (x1_avg, y1_avg, x2_avg, y2_avg)
            print(f'{"average line slope:", mean_slope, "x1:", x1_avg, "y1:", y1_avg, "x2:", x2_avg, "y2:", y2_avg}')

            # Calcular el centro de la línea
            cx = (x1_avg + x2_avg) // 2

            if abs(mean_slope) > 0.5:  # Ajustar los límites según sea necesario
                print("[✓] Línea detectada con pendiente suficiente:", mean_slope)
                steering_needed = True

        # ----- CALCULAR OFFSET -----
        found = best_line is not None

    if found:
        #Dibujamos la línea y punto central que detectamos
        x1, y1, x2, y2 = best_line
        cv2.line(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(display_img, (cx, h - 10), 6, (255, 0, 0), -1)
    
    if steering_needed:
        offset = (cx - (w // 4))  # Calculamos el offset respecto al centro de la imagen
    else:
        offset = 0

    return offset, display_img, steering_needed, found

#Función para montrar la imagen ya procesada en el display de Webots
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

#Función principal
def main():
    #Crear el robot (coche) y el conductor.
    robot = Car()
    driver = Driver()
    last_capture_time = time.time()

    #Obtener el tiempo base
    timestep = int(robot.getBasicTimeStep())

    #Activar la cámara
    camera = robot.getDevice("camera")
    camera.enable(timestep)  #timestep

    #Agregamos la pantalla de visulización
    display_img = Display("display")

    #Configuramos la velocidad deseada 
    speed = 50
    driver.setCruisingSpeed(speed)

    #Parámetros de control
    Kp = 0.005  #Control proporcional más suave
    alpha = 0.3 #Suavizado del giro
    steering_prev = 0.0 #Ultimo valor de giro

    #captura de imagenes

    IMAGE_DIR = "proyecto_final/captured_images"
    CSV_FILE = "proyecto_final/driving_log.csv"
    CAPTURE_INTERVAL = 0.5  # segundos entre capturas
    # Crear carpeta si no existe
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    # Si el CSV no existe, crear y escribir encabezado
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "steering_angle"])


    #Se ejecuta cada timestep
    while robot.step() != -1:

        #Captura y procesa la imagen
        image = get_image(camera)
        offset, processed_image, steering_ned, found = process_image(image)

        #Control proporcional
        if steering_ned:
            steering_raw = Kp * offset #Ajuste proporcional
            print(f"[✓] Offset: {offset}, Steering_raw: {steering_raw:.3f}")
        elif found:   
            steering_raw = 0.0 #Si no hay línea o no es encesario girar, determinamos seguir recto
            print("[x] Linea recta - Manteniendo rumbo recto")
        else:
            steering_raw = 0.0
            print("[x] No se detectó línea - Manteniendo rumbo recto")

        #Aplicamos un filtro exponencial para suavizar los giros brucos
        steering = alpha * steering_raw + (1 - alpha) * steering_prev
        steering_prev = steering

        #Limitamos el valor del giro para evitar giros extremos
        steering = np.clip(steering, -0.5, 0.5)

        #Aplicamos los comandos de giro
        driver.setSteeringAngle(steering)

        #Mostramos la imagen procesada en la pantalla
        display_image(display_img, processed_image)

        # Captura automática
        current_time = time.time()
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_name = f"{timestamp}.png"
            image_path = os.path.join(IMAGE_DIR, image_name)
            camera.saveImage(image_path, 1)

            # Guardar en CSV
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([image_name, steering])
            print(f"[✓] Imagen capturada: {image_name} | Ángulo: {steering:.2f}")
            last_capture_time = current_time

#Entrada
if __name__ == "__main__":
    main()