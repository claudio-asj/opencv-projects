import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def main():
    # Tente mudar para 0, 1 ou 2 se não abrir a câmera
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erro ao abrir a câmera")
        return

    # Usar MediaPipe Holistic para detectar rosto, mãos e corpo
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Criar janela full screen
    window_name = "MediaPipe Holistic - Pressione 1 para sair"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espelhar a imagem para efeito "espelho"
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectar landmarks
        results = holistic.process(frame_rgb)

        # Desenhar landmarks no frame original
        annotated_frame = frame.copy()

        # Rosto
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )

        # Corpo
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
            )

        # Mãos (esquerda)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )

        # Mãos (direita)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )

        # Agora criar imagem preta para desenhar só as landmarks detectadas (copy ampliada)
        height, width, _ = frame.shape
        side_image = np.zeros_like(frame)

        # Para "copiar" e ampliar, vamos pegar os pontos das landmarks e desenhar só eles numa escala maior

        # Função auxiliar para desenhar landmarks ampliados
        def draw_landmarks_scaled(landmarks, connections, color, thickness, circle_radius, scale=2.0):
            if landmarks is None:
                return
            # landmarks são normalizados (0-1), pegar coordenadas pixel e multiplicar scale
            for landmark in landmarks.landmark:
                x_px = int(landmark.x * width)
                y_px = int(landmark.y * height)

                # coordenadas ampliadas
                x_s = int(x_px * scale)
                y_s = int(y_px * scale)

                # Para não sair da tela (clamp)
                if 0 <= x_s < width and 0 <= y_s < height:
                    cv2.circle(side_image, (x_s, y_s), circle_radius*2, color, -1)

            # Desenhar conexões com linhas ampliadas
            if connections:
                for connection in connections:
                    start = landmarks.landmark[connection[0]]
                    end = landmarks.landmark[connection[1]]
                    x_start = int(start.x * width * scale)
                    y_start = int(start.y * height * scale)
                    x_end = int(end.x * width * scale)
                    y_end = int(end.y * height * scale)

                    if (0 <= x_start < width and 0 <= y_start < height and
                        0 <= x_end < width and 0 <= y_end < height):
                        cv2.line(side_image, (x_start, y_start), (x_end, y_end), color, thickness)

        # Desenhar tudo ampliado na imagem preta
        # IMPORTANTE: O lado da tela tem mesma largura e altura do frame (porque full screen pode variar, mas vamos considerar frame)

        scale = 2.0  # ampliação das landmarks

        # Se quiser deixar lado maior, ajustar a janela ou frame

        # Só desenhar landmarks que existirem
        if results.face_landmarks:
            draw_landmarks_scaled(
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                (0,255,0),
                1,
                1,
                scale
            )
        if results.pose_landmarks:
            draw_landmarks_scaled(
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                (255,0,0),
                2,
                2,
                scale
            )
        if results.left_hand_landmarks:
            draw_landmarks_scaled(
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                (0,0,255),
                2,
                3,
                scale
            )
        if results.right_hand_landmarks:
            draw_landmarks_scaled(
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                (0,0,255),
                2,
                3,
                scale
            )

        # Combinar as duas imagens lado a lado
        combined = np.zeros((height, width*2, 3), dtype=np.uint8)
        combined[:, :width] = annotated_frame
        combined[:, width:] = side_image

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
