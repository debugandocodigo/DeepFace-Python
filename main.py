import cv2
from deepface import DeepFace

# Inicializa o detector de faces do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mapeamento de emoções em inglês para português
emotion_translation = {
    'angry': 'zangado',
    'disgust': 'nojo',
    'fear': 'medo',
    'happy': 'feliz',
    'sad': 'triste',
    'surprise': 'surpreso',
    'neutral': 'normal'
}

# Inicializa a câmera (pode ser necessário ajustar o número do dispositivo)
cap = cv2.VideoCapture(2)

while True:
    try:
        ret, frame = cap.read()  # Captura o frame da webcam
        frame = cv2.flip(frame, 1) # Espelha o frame horizontalmente

        # Converte o frame para escala de cinza para a detecção de faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta faces no frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Itera sobre as faces detectadas
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w] # Recorta a região do rosto

            # Realiza a análise de emoções no rosto recortado
            result = DeepFace.analyze(face_roi, actions=['emotion'])
            emotions = result[0]['emotion']
            dominant_emotion = max(emotions, key=emotions.get) # Encontra a emoção mais predominante

            # Desenha o retângulo ao redor do rosto
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Adiciona o texto da emoção mais predominante ao frame
            text = f"{emotion_translation[dominant_emotion]}"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 2, cv2.LINE_AA)

        # Exibe o frame
        cv2.imshow('Analise de Emocoes', frame)

        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except: pass

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
