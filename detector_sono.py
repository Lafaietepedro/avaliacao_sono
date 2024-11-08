import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import os
import random

def calculo_ear(face, pts_olho_dir, pts_olho_esq):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_esq = face[pts_olho_esq, :]
        face_dir = face[pts_olho_dir, :]

        ear_esq = (np.linalg.norm(face_esq[0] - face_esq[1]) + np.linalg.norm(face_esq[2] - face_esq[3])) / (2 * (np.linalg.norm(face_esq[4] - face_esq[5])))
        ear_dir = (np.linalg.norm(face_dir[0] - face_dir[1]) + np.linalg.norm(face_dir[2] - face_dir[3])) / (2 * (np.linalg.norm(face_dir[4] - face_dir[5])))
    except:
        ear_esq = 0.0
        ear_dir = 0.0
    media_ear = (ear_esq + ear_dir) / 2
    return media_ear

def calculo_mar(face, pts_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_boca = face[pts_boca, :]

        mar = (np.linalg.norm(face_boca[0] - face_boca[1]) + np.linalg.norm(face_boca[2] - face_boca[3]) + np.linalg.norm(face_boca[4] - face_boca[5])) / (2 * (np.linalg.norm(face_boca[6] - face_boca[7])))
    except:
        mar = 0.0
    return mar

audios_rosto = ["bemtevi.mp3", "carcara.mp3", "joaodebarro.mp3", "sabia.mp3", "seriema.mp3"]

audios_rosto_nomes = {
    "bemtevi.mp3": "bem-te-vi",
    "carcara.mp3": "carcara",
    "joaodebarro.mp3": "joao-de-barro",
    "sabia.mp3": "sabia",
    "seriema.mp3": "seriema"
}

pygame.mixer.init()

audio_boca = "foguete.mp3"

pts_olho_esq = [385, 380, 387, 373, 362, 263]
pts_olho_dir = [160, 144, 158, 153, 33, 133]
pts_olhos = pts_olho_esq + pts_olho_dir

pts_boca = [82, 87, 13, 14, 312, 317, 78, 308]

ear_limiar = 0.25
mar_limiar = 0.25
dormindo = 0

camera = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

som_tocando = False
ultimo_tempo_audio = time.time()

som_tocando_boca = False
ultimo_tempo_audio_boca = time.time()

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while camera.isOpened():
        sucesso, frame = camera.read()
        if not sucesso:
            print('Ignorando o frame vazio da câmera.')
            continue
        
        comprimento, largura, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        saida_facemesh = facemesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


        if saida_facemesh.multi_face_landmarks:
            print("Rosto detectado")
            tempo_atual = time.time()
            if tempo_atual - ultimo_tempo_audio >= 10 and som_tocando == False:
                audio_aleatorio = random.choice(audios_rosto)
                try:
                    pygame.mixer.music.load(audio_aleatorio)
                    pygame.mixer.music.play()
                    ultimo_tempo_audio = tempo_atual
                    nome_audio = audios_rosto_nomes.get(audio_aleatorio, "desconhecido")
                except pygame.error as e:
                    print(f"Erro ao carregar o áudio {audio_aleatorio}: {e}")
        else:
            print("Nenhum rosto detectado")
            pygame.mixer.music.stop()
            som_tocando = False

        try:
            for face_landmarks in saida_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 102, 102), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 204, 0), thickness=1, circle_radius=1)
                )
                
                face = face_landmarks.landmark
                
                for id_coord, coord_xyz in enumerate(face):
                    if id_coord in pts_olhos:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, largura, comprimento)
                        cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)
                    if id_coord in pts_boca:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, largura, comprimento)
                        cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)

                ear = calculo_ear(face, pts_olho_dir, pts_olho_esq)
                cv2.rectangle(frame, (0, 1), (312, 210), (58, 58, 55), -1)
                cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.9, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Audio: {nome_audio}", (1, 110),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.9, (255, 255, 255), 2)

                mar = calculo_mar(face, pts_boca)
                cv2.putText(frame, f"MAR: {round(mar, 2)}", (1, 50),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {round(mar, 2)}", (1, 50),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.9, (255, 255, 255), 2)
                cv2.putText(frame,f"{ 'Boca aberta' if mar >= mar_limiar else  'Boca fechada '}", (1, 50),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.9, (255, 255, 255), 2)

                if ear < ear_limiar:
                    t_inicial = time.time() if dormindo == 0 else t_inicial
                    dormindo = 1
                if dormindo == 1 and ear >= ear_limiar:
                    dormindo = 0
                t_final = time.time()
                
                if mar < mar_limiar:
                    try:
                        pygame.mixer.music.load(audio_boca)
                        pygame.mixer.music.play()
                        ultimo_tempo_audio_boca = tempo_atual
                    except pygame.error as e:
                        print(f"Erro ao carregar o áudio {audio_aleatorio}: {e}")

                tempo = (t_final - t_inicial) if dormindo == 1 else 0.0
                cv2.putText(frame, f"Tempo: {round(tempo, 3)}", (1, 80),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.9, (255, 255, 255), 2)
                if tempo >= 1.5:
                    cv2.rectangle(frame, (30, 400), (610, 452), (109, 233, 219), -1)
                    cv2.putText(frame, f"Muito tempo com olhos fechados!", (80, 435),
                                cv2.FONT_HERSHEY_TRIPLEX,
                                0.85, (58, 58, 55), 1)

        except Exception as e:
            print("Erro:", e)

        finally:
            print("Processamento concluído")
        
        cv2.imshow('Camera', frame)
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

camera.release()
cv2.destroyAllWindows() 