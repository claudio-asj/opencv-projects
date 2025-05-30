import cv2
import numpy as np

class SimpleFaceDetector:
    def __init__(self):
        # Carregar classificadores Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def draw_labeled_rectangle(self, img, x, y, w, h, label, color):
        """Desenha retângulo com label"""
        # Desenhar retângulo
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Preparar texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Calcular tamanho do texto
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Posição do texto (acima do retângulo)
        text_x = x
        text_y = y - 10
        
        # Se não couber acima, colocar abaixo
        if text_y < text_size[1]:
            text_y = y + h + text_size[1] + 10
        
        # Desenhar fundo do texto
        cv2.rectangle(img, (text_x - 2, text_y - text_size[1] - 2), 
                     (text_x + text_size[0] + 2, text_y + 2), color, -1)
        
        # Desenhar texto
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    def detect_face_parts(self, frame):
        """Detecta rosto, olhos e estima nariz"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostos
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50)
        )
        
        for (x, y, w, h) in faces:
            # Desenhar rosto
            self.draw_labeled_rectangle(frame, x, y, w, h, "ROSTO", (0, 255, 0))
            
            # Região dos olhos (terço superior do rosto)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detectar olhos apenas na região superior do rosto
            eye_region_height = int(h * 0.6)  # 60% superior do rosto
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray[0:eye_region_height, 0:w],
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(15, 15)
            )
            
            # Processar olhos detectados
            eye_positions = []
            for (ex, ey, ew, eh) in eyes:
                # Coordenadas absolutas
                abs_ex = x + ex
                abs_ey = y + ey
                
                # Guardar posição para calcular o nariz
                eye_positions.append((abs_ex + ew//2, abs_ey + eh//2))
                
                # Desenhar olho
                self.draw_labeled_rectangle(frame, abs_ex, abs_ey, ew, eh, "OLHO", (0, 0, 255))
            
            # Estimar posição do nariz
            # Nariz fica no centro horizontal do rosto e na região média vertical
            nose_x = x + int(w * 0.35)
            nose_y = y + int(h * 0.45)
            nose_w = int(w * 0.3)
            nose_h = int(h * 0.3)
            
            # Se temos olhos detectados, usar sua posição para melhor estimativa
            if len(eye_positions) >= 2:
                # Calcular centro entre os olhos
                center_x = sum([pos[0] for pos in eye_positions]) // len(eye_positions)
                center_y = sum([pos[1] for pos in eye_positions]) // len(eye_positions)
                
                # Nariz um pouco abaixo do centro dos olhos
                nose_x = center_x - nose_w // 2
                nose_y = center_y + int(h * 0.15)
            
            # Desenhar nariz
            self.draw_labeled_rectangle(frame, nose_x, nose_y, nose_w, nose_h, "NARIZ", (255, 0, 255))
        
        return frame

def main():
    print("=== DETECTOR SIMPLES DE PARTES DO ROSTO ===")
    print("Detecta: ROSTO, OLHOS e NARIZ")
    print()
    
    # Inicializar detector
    detector = SimpleFaceDetector()
    
    # Inicializar câmera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERRO: Não foi possível abrir a câmera")
        return
    
    print("Camera iniciada com sucesso!")
    print("Controles:")
    print("  'q' - Sair")
    print("  's' - Salvar captura")
    print("  'ESPACO' - Pausar/Retomar")
    print()
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("ERRO: Erro ao capturar frame")
                break
            
            # Espelhar horizontalmente
            frame = cv2.flip(frame, 1)
            
            # Detectar partes do rosto
            frame = detector.detect_face_parts(frame)
            
            frame_count += 1
        
        # Adicionar informações na tela
        status_color = (0, 255, 0) if not paused else (0, 255, 255)
        status_text = "ATIVO" if not paused else "PAUSADO"
        
        # Informações no canto superior
        info_texts = [
            f"Status: {status_text}",
            "Controles: 'q'=Sair 's'=Salvar ESPAÇO=Pausar",
            f"Frames: {frame_count}"
        ]
        
        for i, text in enumerate(info_texts):
            y_pos = 25 + i * 25
            # Fundo do texto
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (5, y_pos - 20), (text_size[0] + 10, y_pos + 5), (0, 0, 0), -1)
            # Texto
            color = status_color if i == 0 else (255, 255, 255)
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Mostrar frame
        cv2.imshow('Detector Simples - Rosto, Olhos e Nariz', frame)
        
        # Processar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nEncerrando programa...")
            break
        elif key == ord('s'):
            filename = f'captura_rosto_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Captura salva como '{filename}'")
        elif key == ord(' '):
            paused = not paused
            status = "pausado" if paused else "retomado"
            print(f"Video {status}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Programa encerrado com sucesso!")

if __name__ == "__main__":
    main()