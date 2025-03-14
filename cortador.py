# Arquivo usado para gerar fundos aleatórios 

# Importações necessárias
import os
import random
from PIL import Image

# Função para cortar a imagem
def cortar_imagem(imagem_path, imagens_saida):
    # Caso o diretório de saída não exista, ele é criado
    if not os.path.exists(imagens_saida):
        os.makedirs(imagens_saida)

    # Abre a imagem e pega a largura e altura
    imagem = Image.open(imagem_path)
    largura, altura = imagem.size

    # Corta a imagem em 3 partes
    for i in range(3):
        # Gera uma posição aleatória 
        x = random.randint(0, largura - 600)
        y = random.randint(0, altura - 600)
        quadrado = (x, y, x + 600, y + 600) # Área a ser recortada
        recorte = imagem.crop(quadrado) # Recorta a imagem
        recorte.save(os.path.join(imagens_saida, f'background_{i}.png')) # Salva a imagem

if __name__ == "__main__":
    imagens_entrada = 'original' # Pasta com as imagens usadas para gerar os fundos
    imagens_saida = 'entrada//backgrounds' # Pasta onde os fundos vão ser armazenados

    for nome_imagem in os.listdir(imagens_entrada): # for que percorre cada imagem na pasta
        imagem_path = os.path.join(imagens_entrada, nome_imagem)  # Cria o diretório da imagem
        cortar_imagem(imagem_path, imagens_saida) # Chama a função
