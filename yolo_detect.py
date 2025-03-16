import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# define

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Caminho para o arquivo do modelo YOLO (exemplo: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Fonte da imagem, pode ser um arquivo de imagem ("test.jpg"), \
                    pasta de imagens ("test_dir"), arquivo de vídeo ("testvid.mp4"), índice da câmera USB ("usb0") ou índice da Picamera ("picamera0")', 
                    required=True)
parser.add_argument('--thresh', help='Limite mínimo de confiança para exibir objetos detectados (exemplo: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolução em LxA para exibir os resultados da inferência (exemplo: "640x480"), \
                    caso contrário, corresponde à resolução da fonte',
                    default=None)
parser.add_argument('--record', help='Gravar resultados de vídeo ou webcam e salvá-los como "demo1.avi". Deve especificar o argumento --resolution para gravar.',
                    action='store_true')

args = parser.parse_args()


# Analise as entradas do usuário
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# verifica se o modelo existe
if (not os.path.exists(model_path)):
    print('Erro: modelo não encontrado')
    sys.exit(0)

model = YOLO(model_path, task='detect') # carrega o modelo com a tarefa de detecção
labels = model.names # carrega as classes

# entrada de imagem
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'A extensão do arquivo {ext} não é suportada.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'A entrada {img_source} é inválida.')
    sys.exit(0)

# resolução da exibição
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

if record:
    if source_type not in ['video','usb']:
        print('A gravação só funciona para fontes de vídeo e câmera. Por favor, tente novamente.')
        sys.exit(0)
    if not user_res:
        print('Por favor, especifique a resolução para gravar o vídeo.')
        sys.exit(0)
    
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # resolucao
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# variaveis de controle
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

while True:

    t_start = time.perf_counter()

    if source_type == 'image' or source_type == 'folder': 
        if img_count >= len(imgs_list):
            print('As imagens acabaram')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video': #
        ret, frame = cap.read()
        if not ret:
            print('Vídeo acabou')
            break
    
    elif source_type == 'usb':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('camera USB não encontrada ou não funcionando')
            break

    elif source_type == 'picamera': 
        frame = cap.capture_array()
        if (frame is None):
            print('camera não encontrada ou não funcionando')
            break

    # Redimensione o quadro para a resolução de exibição desejada
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    results = model(frame, verbose=False)

    detections = results[0].boxes

    object_count = 0

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze() 
        xmin, ymin, xmax, ymax = xyxy.astype(int) 

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        conf = detections[i].conf.item()

        if conf > 0.5:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) 
            label_ymin = max(ymin, labelSize[1] + 10) 
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) 
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 

            # conta o número de objetos
            object_count = object_count + 1

    # caso seja vídeo, calcula o fps
    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) 
    
    cv2.putText(frame, f'Numero de objetos: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('Resultados da detecção YOLO',frame) # exibe a imagem
    if record: recorder.write(frame)

    # caso esteja passando por uma pasta, ele aguarda o usuário para passar para a proxima
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # q (quit)
        break
    elif key == ord('s') or key == ord('S'): # s (stop)
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # p (print)
        cv2.imwrite('capture.png',frame)
    
    # Calcula os fps para o quadro atual
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Adicione o resultado do FPS ao frame_rate_buffer (para encontrar a média de FPS em vários quadros)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calcula a média dos fps
    avg_frame_rate = np.mean(frame_rate_buffer)


# Limpeza
print(f'FPS médio do pipeline: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()