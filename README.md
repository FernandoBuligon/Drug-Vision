# Trabalho Final - LAMIA
## Resumo
Ao longo do bootcamp de Machine Learning do LAMIA, é adquirido diversos conhecimentos. No final, é necessário desenvolver um projeto para aplicar o que foi aprendido em uma área escolhida. No meu caso, optei por visão computacional, trabalhando com um modelo de classificação e outro de detecção, ambos utilizando o YOLO11.
## Sobre o LAMIA
O [LAMIA](https://www.lamia-edu.com) é um laboratório de aprendizado de máquina aplicado à indústria, presente na UTFPR Campus Santa Helena, fundado pelo professor Dr. Thiago França Naves. Tem como missão "liderar o desenvolvimento de pesquisadores e profissionais em Inteligência Artificial que irão guiar o país na direção da inovação através da produção de conhecimento científico, soluções customizadas e produtos para a indústria brasileira e mundial". Trabalhando nas áreas de aprendizado de máquina, ciência de dados e tecnologias imersivas.

![Imagem não carregada](https://www.lamia-edu.com/_next/image?url=%2Fimages%2Ficon-novaLogo.png&w=384&q=75)
## Base de dados - Classificação
O primeiro passo dado foi buscar alguma base de dados que eu pudesse utilizar. Consegui encontrar uma que me chamou atenção através do [Kaggle](https://www.kaggle.com), se chama [Pharmaceutical Drugs and Vitamins Synthetic Images](https://www.kaggle.com/datasets/vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images), ele conta com 10.000 imagens sintéticas e 10 classes diferentes de remédios populares na Filipinas.
Ao decorrer do desenvolvimento, fiz algumas alterações:
- Diminuição do tamanho: Como são pílulas, elas não possuem variações dentro de suas próprias classes, a única coisa que varia são os fundos das imagens, mas o elemento principal não, por isso cortei 80% dos elementos, ficando com apenas 200 imagens por classe. Para compensar, aumentei levemente o número de épocas, tendo menos imagens para processar, mais rápido é o treinamento. No final, obtive resultados melhores em menos tempo.
- Adição de classes: Todas as imagens são sintéticas, então pensei em adicionar minhas próprias classes. Para isso, tirei foto de algumas pílulas, cortei cada uma através do [GIMP](https://www.gimp.org), salvando em formato png. Para gerar os fundos, tirei fotos de cantos aleatórios com meu próprio celular, e fiz um algoritmo em python ([cortador.py](https://github.com/FernandoBuligon/Drug-Vision/cortador.py)) que passa por cada foto e recorta três quadrados aleatórios de 600x600. Tendos os fundos e as pílulas, entrei em contato com [Lanz Vincent](https://www.linkedin.com/in/lanz-vincent-ds/), criador do dataset, para saber como eu poderia gerar as imagens no mesmo estilo, ele me passou o algoritmo que usou, pertence a [Adam Kelly](https://github.com/akTwelve) e está disponível em um dos seus repositórios, chamado [cocosynth](https://github.com/akTwelve/cocosynth/tree/master). No final foram adicionadas 8 classes, totalizando 18.
- Troca de formato: Foi necessário deixar no formato requisitado pelo YOLO11, que seria uma pasta com um nome qualquer, dentro dela uma pasta test, train e val (opcional), dentro dessas três pastas tem que ter uma pasta para cada classe nomeada com seu respectivo nome.
## Base de dados - Detecção
Para o modelo de detecção, foi necessário usar o [Label Studio](https://labelstud.io) para rotular cada pílula, como o processo é manual, ele acaba não sendo muito rápido dependendo do número de imagens, mas é bem simples. Depois foi só exportar o projeto no formato "YOLO with Images" e está pronto para ser usado.
## Modelo
Tendo em mãos a base de dados, foi escolhido o modelo pre treinado [YOLO11](https://docs.ultralytics.com/pt/models/yolo11/), decidi usar ele por ter uma documentação limpa, ser simples e eficiente de usar, e conta com tutoriais da própria Ultralytics. Também apresenta melhorias em relação aos modelos anteriores.

![Imagem não carregada](https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png)

## Resultados
Em ambos os casos foram obtidos resultados satisfatórios
