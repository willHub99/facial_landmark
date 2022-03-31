
# Sistema de Monitoramento de fadiga usando Python

A partir da captura da face do usuário pela câmera do seu computador é realizado um mapeamento facial
dos principais pontos de referência facial utilizando a biblioteca dlib, esta que utiliza-se de modelos 
pré treinados. 


## Autores
- Orientador: Daniel Cavalcanti Jeronymo [Página pessoal](https://coenc.td.utfpr.edu.br/~danielc/)
- [@willianrodrigo](https://github.com/willHub99)


## Instalação de bibliotecas

Clone o projeto

```bash
  git clone https://github.com/willHub99/facial_landmark.git
```

Entre no diretório do projeto

```bash
  cd your-path/controller_webcam
```

Instale as dependências

```bash
  pip install numpy
```

```bash
  pip install scipy
```

```bash
  pip install opencv-python
```

```bash
  pip install pickle5
```

```bash
  pip install imutils
```

```bash
  pip install matplotlib
```

### instalar a biblioteca dlib

  Instalar o CMake: https://cmake.org/download/
  Verificar o o CMake foi adicionado ao path das variáveis de ambiente do usuário


  Instalar o Visual Studio: https://visualstudio.microsoft.com/pt-br/downloads/
  Habilitar pacotes adicionais para programação C, C++ (Desktop development with C++)


```bash
  pip install cmake
```

### Opções de instalação da biblioteca dlib

- Opção 1: Nome do Pacote
```bash
  pip install dlib
```

- Opção 2: Repositório do GitHub
```bash
  pip install git+https://github.com/davisking/dlib
```

- Opção 3: Arquivo compactado (.zip)
Baixe o arquivo compactado: https://github.com/davisking/dlib
```bash
  pip install your-path-directory/dlib-19.23.0.zip
```
    
## Rodando localmente


Inicie a Aplicação

```bash
  python main.py
```


## Screenshots

- Captura da câmera

<p align="center">
  <img src="">
</p>

- Captura da câmera - Incidente

<p align="center">
  <img src="">
</p>
## Demonstração

<p align="center">
  <img src="">
</p>




## Referência

 - [Mapeamento facial (landmarks) com Dlib + python](https://medium.com/brasil-ai/mapeamento-facial-landmarks-com-dlib-python-3a200bb35b87)
 - [Como Controlar WebCam com Python - Introdução ao OpenCV](https://www.youtube.com/watch?v=r8Qg3NfdiHc&ab_channel=HashtagPrograma%C3%A7%C3%A3o)
 - [Eye blink detection with OpenCV, Python, and dlib](https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)