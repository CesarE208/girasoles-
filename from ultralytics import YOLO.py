from ultralytics import YOLO
import cv2 
model = YOLO("yolov8n.pt") #Tipo de modelo
cap = cv2.VideoCapture(0) #Reconocer la camara como entrada de datos
while True:
    ret, frame = cap.read()
    results = model(frame)
    anotaciones_pFrame = results[0].plot()
    cv2.imshow("Yolov8", anotaciones_pFrame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()