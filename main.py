import cv2

class FaceDetector:
    def __init__(self, camera_index=0):
        self.face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        if self.face_cascade_frontal.empty() or self.face_cascade_profile.empty():
            raise Exception("No se pudieron cargar los clasificadores haarcascade.")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("No se pudo abrir la cámara.")

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados.")

    def detectar_caras(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detección de caras frontales
        faces_frontal = self.face_cascade_frontal.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        # Detección de caras laterales izquierdas
        faces_left = self.face_cascade_profile.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        # Detección de caras laterales derechas (invierte la imagen para detectar el lado derecho)
        gray_flipped = cv2.flip(gray, 1)
        faces_right = self.face_cascade_profile.detectMultiScale(gray_flipped, scaleFactor=1.3, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        # Corrige las coordenadas de las caras derechas
        for (x, y, w, h) in faces_right:
            x = frame.shape[1] - x - w
        return faces_frontal, faces_left, faces_right

    def dibujar_rectangulos(self, frame, faces, color=(255, 0, 0)):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    def ejecutar(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("No se pudo capturar el fotograma.")
                break
            faces_frontal, faces_left, faces_right = self.detectar_caras(frame)
            # Dibuja rectángulos para caras frontales (azul)
            self.dibujar_rectangulos(frame, faces_frontal, color=(255, 0, 0))
            # Dibuja rectángulos para caras laterales izquierdas (verde)
            self.dibujar_rectangulos(frame, faces_left, color=(0, 255, 0))
            # Dibuja rectángulos para caras laterales derechas (rojo)
            self.dibujar_rectangulos(frame, faces_right, color=(0, 0, 255))
            cv2.imshow('Face detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    face_detector = FaceDetector()
    face_detector.ejecutar()
