# Nanodegree Engenheiro de Machine Learning
## Projeto final
Leandro Humberto Vieira
11 de maio de 2018

## Introdução

Este repositório contém os resultados do projeto final do NanoDegree Engenheiro de Machine Learning. O projeto consiste na criação de uma aplicação web que permite a classificação de imagens de roupas. Para uma explicação detalhada do problema e da solução, consultar o [relatório de conclusão do projeto](https://github.com/leandrohmvieira/ML-Capstone-Project/blob/master/Docs/Report.md).

## Estrutura do Projeto

* **ML-Capstone-Project** - _Raiz do projeto_

  * **Docs** - _Documentação geral do projeto, contém os relatórios de proposta e conclusão e o readme do conjunto de dados utilizado_

  * **Images** - _Imagens utilizadas no relatório de conclusão_

  * **Notebooks** - _Contém os Python notebooks com o código utilizado para a arquitetura e treinamentos das redes neurais empregadas na solução_

  * **Source** - _Contém todo o código do Framework web Flask, utilizado para executar a interface web da aplicação_

## Instruções de uso

**Importante: As instruções de uso descrevem os passos necessários para a execução do servidor no ambiente [paperspace](https://www.paperspace.com/), utilizado para o desenvolvimento desta solução**

## 1. Criação do ambiente

Para criar um ambiente no Paperspace é necessário criar uma conta de usuário, o cadastro pode ser feito utilizando o link abaixo:

* https://www.paperspace.com/&R=ZPF9F3P

**Utilizando o meu link de referência para o cadastro te garante U$10 de crédito**, o necessário para executar a aplicação deste projeto.

Após ter criado a conta no paperspace, deve-se criar uma instância para a execução do servidor, a instância utilizada na execução desde projeto é o public template fast.ai, equipado com uma [Nvidia Quadro P4000](https://nvidiastore.com.br/nvidia-quadro-p4000). Criar uma instância com GPU requere uma etapa de aprovação do paperspace, então fica a critério a utilização de uma instância sem GPU, que pode ser criada instantâneamente.
Para executar o servidor flask na instância, também é necessário a habilitação de um IP público para acessá-lo pela internet.

Para habilitar o IP público, basta clicar no checkbox correspodente no momento da criação da instância.

## 2. Configuração do Projeto

### 2.1 github
Após ter a instância em funcionamento, conecte ao servidor via SSH (o paperspace te envia um email com detalhes e senhas para fazê-lo) e baixe o projeto do github utilizando o comando:

 `git clone https://github.com/leandrohmvieira/ML-Capstone-Project.git`

### 2.2 CUDA
Caso a instância iniciada seja uma instância com GPU, é necessária a instalação do NVIDIA CUDA 9.0 para a execução do tensorflow. Para instalar os drivers necessários, execute os comandos abaixo:

```
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.4-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libnccl2_2.1.4-1+cuda9.0_amd64.deb
sudo dpkg -i libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
sudo apt-get update
sudo apt-get install cuda=9.0.176-1
sudo apt-get install libcudnn7-dev
sudo apt-get install libnccl-dev
```
**Reinicie a instância e execute os comandos abaixo:**
```
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Caso a instância não tenha GPU, este passo não é necessário, porém o arquivo [requirements.txt](https://github.com/leandrohmvieira/ML-Capstone-Project/blob/master/Source/requirements.txt) deve ser alterado para o tensorflow comum.

## 2.3 Dependências

A solução desenvolvida usa uma série de pacotes Python, para realizar a instalação das bibliotecas navegue até a pasta `Source` e execute o comando:

`pip install -r requirements.txt`

## 2.4 Modelo de rede neural

Para o servidor web funcionar, é necessário que o modelo de rede neural seja baixado também, como ele é muito grande, o mesmo está hospedado em minha conta no google drive. Para baixá-lo, navegue para a pasta `/Source/models` e execute o comando:

 `python download_model.py 1YD2SH2xT0D4dMtI5kcnDWdT791x9YKbU basicCNN.h5`

## 2.5 Porta web

A instância tem suas portas fechadas para acessos externos por padrão, como iremos executar o servidor flask na porta 5000, devemos abrir esta porta no sistema operacional para receber acessos externos. Para realizar esta operação, execute o comando:

`sudo ufw allow 5000`

# 3. Execução

 Após todos os passos configurados, basta executar o servidor Python. Para isso navegue até a pasta `Source` e execute o comando:

 `python app.py`

 Se tudo ocorreu normalmente, em alguns segundos será possível acessar a aplicação pela internet através do endereço `<ippublicodainstancia>:5000`

# 4. Referências

[Tutorial Keras rest api](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)

[Siraj - how to deploy a keras model to production](https://www.youtube.com/watch?v=f6Bf3gl4hWY&t=47s)

[Repositório Siraj - how to deploy a keras model to production](https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production)

[Front-end Deep Learning Rest API](https://github.com/mtobeiyf/keras-flask-deploy-webapp)
