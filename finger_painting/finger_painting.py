import cv2
import numpy as np
from collections import deque

class FingerPainter:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.canvas = None
        self.drawing = False
        self.last_point = None
        self.no_detection_frames = 0
        self.max_no_detection = 10
        
        # Configurações de desenho
        self.draw_color = (255, 0, 0)  # Azul em BGR
        self.line_thickness = 5
        
        # Valores HSV iniciais
        self.h_min = 0
        self.h_max = 30
        self.s_min = 30
        self.s_max = 150
        self.v_min = 60
        self.v_max = 255
        
        # Detector de rosto
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Controle da interface
        self.show_controls = True
        self.control_height = 200
        
    def create_control_panel(self, width):
        """Cria painel de controles HSV"""
        panel = np.zeros((self.control_height, width, 3), dtype=np.uint8)
        
        # Título
        cv2.putText(panel, "CONTROLES HSV - Ajuste ate ver sua mao", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Barras de controle simuladas
        bar_width = width - 100
        bar_height = 15
        y_positions = [45, 70, 95, 120, 145, 170]
        labels = ['H Min', 'H Max', 'S Min', 'S Max', 'V Min', 'V Max']
        values = [self.h_min, self.h_max, self.s_min, self.s_max, self.v_min, self.v_max]
        max_vals = [179, 179, 255, 255, 255, 255]
        
        for i, (label, value, max_val, y_pos) in enumerate(zip(labels, values, max_vals, y_positions)):
            # Label
            cv2.putText(panel, f"{label}:", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Barra de fundo
            cv2.rectangle(panel, (80, y_pos - 10), (80 + bar_width, y_pos), (50, 50, 50), -1)
            
            # Barra preenchida
            fill_width = int((value / max_val) * bar_width)
            cv2.rectangle(panel, (80, y_pos - 10), (80 + fill_width, y_pos), (0, 255, 0), -1)
            
            # Valor
            cv2.putText(panel, str(value), (80 + bar_width + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instruções
        cv2.putText(panel, "Use as teclas:", (width - 200, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(panel, "1,2: H Min/Max", (width - 200, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel, "3,4: S Min/Max", (width - 200, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel, "5,6: V Min/Max", (width - 200, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel, "+ Shift: aumenta", (width - 200, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel, "Sem Shift: diminui", (width - 200, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel, "'h': mostra/oculta", (width - 200, 165), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return panel
    
    def detect_faces(self, frame):
        """Detecta rostos na imagem"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def is_point_in_face(self, point, faces):
        """Verifica se um ponto está dentro de algum rosto detectado"""
        for (x, y, w, h) in faces:
            if x <= point[0] <= x + w and y <= point[1] <= y + h:
                return True
        return False
    
    def create_skin_mask(self, hsv, faces):
        """Cria máscara para detecção de pele, excluindo rostos"""
        # Cria máscara com os valores HSV atuais
        lower_skin = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
        upper_skin = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Remove áreas dos rostos da máscara
        for (x, y, w, h) in faces:
            # Expande um pouco a área do rosto para garantir
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(mask.shape[1], x + w + margin)
            y2 = min(mask.shape[0], y + h + margin)
            mask[y1:y2, x1:x2] = 0
        
        # Processamento da máscara
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Operações morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
    
    def get_largest_contour(self, contours):
        """Retorna o maior contorno por área"""
        if len(contours) == 0:
            return None
        return max(contours, key=cv2.contourArea)
    
    def get_fingertip(self, contour):
        """Encontra a ponta do dedo (ponto mais alto)"""
        # Ponto mais alto do contorno
        top_point = tuple(contour[contour[:, :, 1].argmin()][0])
        return top_point
    
    def is_valid_point(self, point, faces):
        """Verifica se o ponto é válido para desenho"""
        # Verifica se não está em um rosto
        if self.is_point_in_face(point, faces):
            return False
            
        if self.last_point is None:
            return True
        
        # Calcula distância do último ponto
        distance = np.sqrt((point[0] - self.last_point[0])**2 + 
                          (point[1] - self.last_point[1])**2)
        
        return distance < 80
    
    def draw_interface(self, frame, faces, hand_area=0):
        """Desenha interface principal"""
        height, width = frame.shape[:2]
        
        # Desenha rostos detectados
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(frame, "ROSTO IGNORADO", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Status na parte superior
        status = "Desenhando" if self.drawing else "Procurando mao"
        status_color = (0, 255, 0) if self.drawing else (0, 255, 255)
        
        # Fundo para status
        cv2.rectangle(frame, (5, 5), (400, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (400, 80), (255, 255, 255), 2)
        
        cv2.putText(frame, f"Status: {status}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        if hand_area > 0:
            cv2.putText(frame, f"Area da mao: {int(hand_area)}", (10, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(frame, f"Rostos detectados: {len(faces)}", (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Instruções na parte inferior
        instructions = [
            "'c': Limpar canvas",
            "'q': Sair programa", 
            "'h': Mostrar/ocultar controles",
            "1-6: Ajustar HSV (+ Shift para aumentar)"
        ]
        
        start_y = height - 80
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, start_y + i * 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def process_key_input(self, key):
        """Processa entrada do teclado para ajustar HSV"""
        step = 5
        
        # Verifica se Shift está pressionado (valores maiores indicam Shift)
        if key >= 65 and key <= 90:  # Letras maiúsculas = Shift pressionado
            increase = True
            key = key + 32  # Converte para minúscula
        else:
            increase = False
        
        # Ajusta valores HSV
        if key == ord('1'):
            self.h_min = max(0, self.h_min - step) if not increase else min(179, self.h_min + step)
        elif key == ord('2'):
            self.h_max = max(0, self.h_max - step) if not increase else min(179, self.h_max + step)
        elif key == ord('3'):
            self.s_min = max(0, self.s_min - step) if not increase else min(255, self.s_min + step)
        elif key == ord('4'):
            self.s_max = max(0, self.s_max - step) if not increase else min(255, self.s_max + step)
        elif key == ord('5'):
            self.v_min = max(0, self.v_min - step) if not increase else min(255, self.v_min + step)
        elif key == ord('6'):
            self.v_max = max(0, self.v_max - step) if not increase else min(255, self.v_max + step)
    
    def run(self):
        """Loop principal do programa"""
        print("=== Finger Painter Avançado ===")
        print("Recursos:")
        print("- Detecção de rosto (ignora rostos)")
        print("- Interface unificada")
        print("- Controles HSV integrados")
        print("\nControles:")
        print("- Teclas 1-6: Ajustar HSV")
        print("- Shift + tecla: Aumentar valor")
        print("- Sem Shift: Diminuir valor")
        print("- 'h': Mostrar/ocultar controles")
        print("- 'c': Limpar canvas")
        print("- 'q': Sair")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Erro ao capturar frame da câmera")
                break
            
            # Espelha a imagem
            frame = cv2.flip(frame, 1)
            original_height, width = frame.shape[:2]
            
            # Inicializa canvas
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)
            
            # Detecta rostos
            faces = self.detect_faces(frame)
            
            # Converte para HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Cria máscara excluindo rostos
            mask = self.create_skin_mask(hsv, faces)
            
            # Encontra contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hand_contour = self.get_largest_contour(contours)
            
            hand_area = 0
            current_point = None
            
            # Processa detecção da mão
            if hand_contour is not None:
                hand_area = cv2.contourArea(hand_contour)
                
                if hand_area > 2000:
                    # Desenha contorno da mão
                    cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
                    
                    # Encontra ponta do dedo
                    fingertip = self.get_fingertip(hand_contour)
                    
                    # Verifica se o ponto é válido (não está em rosto)
                    if self.is_valid_point(fingertip, faces):
                        current_point = fingertip
                        
                        # Desenha círculo na ponta do dedo
                        cv2.circle(frame, current_point, 8, (0, 0, 255), -1)
                        
                        # Desenha linha se estava desenhando
                        if self.drawing and self.last_point is not None:
                            cv2.line(self.canvas, self.last_point, current_point, 
                                    self.draw_color, self.line_thickness)
                        
                        self.last_point = current_point
                        self.drawing = True
                        self.no_detection_frames = 0
                    else:
                        self.no_detection_frames += 1
                else:
                    self.no_detection_frames += 1
            else:
                self.no_detection_frames += 1
            
            # Para de desenhar se não detectar por muito tempo
            if self.no_detection_frames > self.max_no_detection:
                self.drawing = False
                self.last_point = None
            
            # Combina frame com canvas
            result = cv2.add(frame, self.canvas)
            
            # Desenha interface principal
            self.draw_interface(result, faces, hand_area)
            
            # Cria imagem final com controles se necessário
            if self.show_controls:
                # Cria painel de controles
                control_panel = self.create_control_panel(width)
                
                # Combina imagem principal com controles
                final_image = np.vstack([result, control_panel])
                
                # Adiciona máscara ao lado
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask_resized = cv2.resize(mask_colored, (width//3, original_height + self.control_height))
                
                # Redimensiona imagem principal
                main_resized = cv2.resize(final_image, (2*width//3, original_height + self.control_height))
                
                # Combina horizontalmente
                display_image = np.hstack([main_resized, mask_resized])
                
                # Labels
                cv2.putText(display_image, "PRINCIPAL", (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, "MASCARA", (2*width//3 + 10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                display_image = result
            
            cv2.imshow("Finger Painter - Interface Completa", display_image)
            
            # Processa teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                self.canvas = np.zeros_like(frame)
                print("Canvas limpo!")
            elif key == ord('q'):
                break
            elif key == ord('h'):
                self.show_controls = not self.show_controls
                print(f"Controles: {'Visível' if self.show_controls else 'Oculto'}")
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')] or \
                 key in [ord('!'), ord('@'), ord('#'), ord('$'), ord('%'), ord('^')]:
                self.process_key_input(key)
        
        # Limpeza
        self.cap.release()
        cv2.destroyAllWindows()
        print("Programa encerrado.")

# Execução
if __name__ == "__main__":
    painter = FingerPainter()
    painter.run()