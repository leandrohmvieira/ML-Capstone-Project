# Nanodegree Engenheiro de Machine Learning
## Projeto final
Leandro Humberto Vieira
11 de maio de 2018

## I. Definição
A definição deste projeto consiste em construir um método de multiclassificação de roupas, através das técnicas de Deep Learning.

### Visão geral do projeto

A imagem é a primeira impressão que uma pessoa tem ao ver alguém, é algo que nosso cérebro faz automaticamente para armazenar informações e é realizado a partir do momento em que começamos a interagir com esta pessoa. Uma imagem diferente do que uma pessoa é durante uma conversa causa um "ruído" na comunicação e uma confusão no nosso cérebro.

Hoje a imagem é uma via de comunicação pessoal tão importante quanto a fala, dessa forma surgiu a necessidade de profissionais que ensinam a arte da comunicação não-verbal, através de métodos e análises que permitem ensinar uma pessoa a se conhecer e, assim, expressar melhor seu estilo e personalidade através de sua imagem. O processo de [consultoria de imagem](https://en.wikipedia.org/wiki/Image_consulting) ensina a pessoa a identificar seu estilo e sua personalidade, e após este passo, trabalhar com o objetivo de construir uma imagem pessoal mais positiva e coerente com ela mesma.
A consultoria de imagem trabalha em sua grande parte com roupas, logo há algumas etapas do processo consultivo que visam limpar, organizar e classificar as roupas da cliente.

#### Por que a consultoria de imagem é importante

A imagem, o estilo e as vestimentas são fatores que regem ou influenciam vários aspectos da vida de uma pessoa, o que torna esta informação muito valiosa e digna de vários estudos como os descritos abaixo:

* [Como a roupa afeta sua consciência](http://www.dailymail.co.uk/sciencetech/article-2644076/You-DRESS-Clothing-significant-effect-self-esteem-confidence-claims-expert.html) - [Livro descrito no artigo](https://www.amazon.com/Mind-What-You-Wear-Psychology-ebook/dp/B00KBTB3NS/ref=sr_1_1?ie=UTF8&qid=1454635783&sr=8-1&keywords=Mind+What+You+Wear)

* [Como a roupa impacta no seu sucesso profissional](http://www.businessinsider.com/how-your-clothing-impacts-your-success-2014-8)
* [A correlação da vestimenta com seu relacionamento afetivo](https://www.meetmindful.com/the-way-you-dress/)
* [Benefícios da organização das roupas](https://www.janeisatomas.com.br/6-vantagens-de-ter-um-closet-organizado/)

#### Proposta

Este projeto demonstra a tentativa de montar um serviço de classificação de roupas, através da utilização de tecnologia e inteligência artificial para ajudar as pessoas a conhecerem suas roupas, pois saber o que se veste é uma excelente forma de adquirir auto conhecimento. Automatizar a classificação de roupas também seria uma forma de melhorar o trabalho de [consultoria de imagem](https://en.wikipedia.org/wiki/Image_consulting), possibilitando a geração de informações gerenciais de roupas, facilitando o processo de decisão destes profissionais durante a consultoria de um cliente.

[!arquitetura de programa de aprendizado de máquina](colocar aqui imagem de entrada de dados e saída de solução)

Para ensinar as redes neurais a classificarem roupas, foram encontrados na internet alguns conjuntos de imagens que são excelentes candidatos para a solução do problema, os mesmos serão discutidos em detalhes posteriormente.

### Descrição do problema

O problema a ser resolvido se define em como ter uma visão estratégica do armário de uma mulher. Apenas olhando para o armário, por mais organizado que esteja, não é possível responder perguntas como:

* Quantas blusas pretas eu tenho?
* Quais são as peças essenciais que faltam no armário?
* Qual a proporção casual/trabalho que tenho?

Para responder tais perguntas, é necessário classificar as roupas entre vários tipos de vestimenta, como alguns destes exemplos abaixo:

1. Lingeries
2. Acessórios
3. Bolsas e sapatos
4. Blusas
5. Terceira peça
6. Calças
7. Shorts
8. Saias
9. Vestidos/Macacões/Macaquinhos
10. Roupas de Academia



Nesta seção, você irá definir o problema que você está tentando resolver de forma clara, incluindo a estratégia (resumo das tarefas) que você irá utilizar para alcançar a solução desejada. Você deverá também discutir detalhadamente qual será a solução pretendida para este problema. Questões para se perguntar ao escrever esta seção:
- _A enunciação do problema foi claramente definida? O leitor irá entender o que você está esperando resolver?_
- _Você discutiu detalhadamente como irá tentar resolver o problema?_
- _A solução antecipada está claramente definida? O leitor entenderá quais resultados você está procurando?_

### Métricas

A métrica de validação a ser utilizada pelo modelo será a [acurácia](https://en.wikipedia.org/wiki/Accuracy_and_precision).

O processo de validação das métricas será dividido nas etapas abaixo:

* Criação de uma [rede neural convolucional](https://pt.wikipedia.org/wiki/Rede_neural_convolucional) completa
* Adaptação de uma rede neural convolucional já construída, aplicando a técnica de [Transfer Learning](https://en.wikipedia.org/wiki/Transfer_learning)
* Treinamento e validação dos modelos
* Comparação da acurácia entre os modelos



## II. Análise
_(aprox. 2-4 páginas)_

### Exploração dos dados
Nesta seção, é esperado que você analise os dados que você está usando para o problema. Esses dados podem ser tanto na forma de um conjunto de dados (ou conjuntos de dados), dados de entrada (ou arquivos de entrada), ou até um ambiente. O tipo de dados deve ser descrito detalhadamente e, se possível, ter estatísticas e informações básicas apresentadas (tais como discussão dos atributos de entrada ou definição de características das entradas ou do ambiente) Qualquer anormalidade ou qualidade interessante dos dados que possam precisar ser devidamente tratadas devem ser identificadas (tais como características que precisem ser transformadas ou a possibilidade de valores atípicos) Questões para se perguntar ao escrever esta seção:
- _Se exite um conjunto de dados para o problema em questão, você discutiu totalmente as características desse conjunto? Uma amostra dos dados foi oferecida ao leitor?_
- _Se existe um conjunto de dados para o problema, as estatísticas sobre eles foram calculadas e reportadas? Foram discutidos quaisquer resultados relevantes desses cálculos?_
- _Se **não** existe um conjunto de dados para o problema, foi realizada uma discussão sobre o espaço de entrada ou os dados de entrada do problema?_
- _Existem anormalidades ou características acerca do espaço de entrada ou conjunto de dados que necessitem ser direcionados? (variáveis categóricas, valores faltando, valores atípicos, etc.)_

### Visualização exploratória
Nesta seção, você precisará fornecer alguma forma de visualização que sintetize ou evidencie uma característica ou atributo relevante sobre os dados. A visualização deve sustentar adequadamente os dados utilizados. Discuta por que essa visualização foi escolhida e por que é relevante. Questões para se perguntar ao escrever esta seção:
- _Você visualizou uma característica ou um atributo relevante acerca do conjunto de dados ou dados de entrada?_
- _A visualização foi completamente analisada e discutida?_
- _Se um gráfico foi fornecido, os eixos, títulos e dados foram claramente definidos?_

### Algoritmos e técnicas

Para realizar a classificação das roupas, serão implementada uma **rede neural**. Redes neurais são conhecidas por atingir o _"estado da arte"_ onde o domínio do problema se referem a imagens, assim ,como primeira opção, foram construídas e treinadas duas **redes neurais convolucionais** para a classificação das imagens.

**Deep Learning** é uma técnica específica de aprendizado de máquina, ou seja, o programa deve "aprender" a solução por si, pois ela não será programada explicitamente. Para isso, o programa utiliza dados de entrada para realizar seu aprendizado, utilizando eles para _"treinar"_.

Nesta seção, você deverá discutir os algoritmos e técnicas que você pretende utilizar para solucionar o problema. Você deverá justificar o uso de cada algoritmo ou técnica baseado nas características do problema e domínio do problema. Questões para se perguntar ao escrever esta seção:
- _Os algoritmos que serão utilizados, incluindo quaisquer variáveis/parâmetros padrão do projeto, foram claramente definidos?_
- _As técnicas a serem usadas foram adequadamente discutidas e justificadas?_
- _Ficou claro como os dados de entrada ou conjuntos de dados serão controlados pelos algoritmos e técnicas escolhidas?_

### Benchmark
Nesta  seção, você deverá definir claramente um resultado de referência (benchmark) ou limiar para comparar entre desempenhos obtidos pela sua solução. O raciocínio por trás da referência (no caso onde não é estabelecido um resultado) deve ser discutido. Questões para se perguntar ao escrever esta seção:
- _Algum resultado ou valor que funcione como referência para a medida de desempenho foi fornecido?_
- _Ficou claro como esse resultado ou valor foi obtido (seja por dados ou por hipóteses)?_


## III. Metodologia

A metodologia aplicada para a solução do problema são redes neurais

### Pré-processamento de dados
Nesta seção, você deve documentar claramente todos os passos de pré-processamento que você pretende fazer, caso algum seja necessário. A partir da seção anterior, quaisquer anormalidades ou características que você identificou no conjunto de dados deverão ser adequadamente direcionadas e tratadas aqui. Questões para se perguntar ao escrever esta seção:
- _Se os algoritmos escolhidos requerem passos de pré-processamento, como seleção ou transformações de atributos, tais passos foram adequadamente documentados?_
- _Baseado na seção de **Exploração de dados**, se existiram anormalidade ou características que precisem ser tratadas, elas foram adequadamente corrigidas?_
- _Se não é necessário um pré-processamento, foi bem definido o porquê?_

### Implementação
Nesta seção, o processo de escolha de quais métricas, algoritmos e técnicas deveriam ser implementados para os dados apresentados deve estar claramente documentado. Deve estar bastante claro como a implementação foi feita, e uma discussão deve ser elaborada a respeito de quaisquer complicações ocorridas durante o processo.  Questões para se perguntar ao escrever esta seção:
- _Ficou claro como os algoritmos e técnicas foram implementados com os conjuntos de dados e os dados de entrada apresentados?_
- _Houve complicações com as métricas ou técnicas originais que acabaram exigindo mudanças antes de chegar à solução?_
- _Houve qualquer parte do processo de codificação (escrita de funções complicadas, por exemplo) que deveriam ser documentadas?_

### Refinamento
Nesta seção, você deverá discutir o processo de aperfeiçoamento dos algoritmos e técnicas usados em sua implementação. Por exemplo, ajuste de parâmetros para que certos modelos obtenham melhores soluções está dentro da categoria de refinamento. Suas soluções inicial e final devem ser registradas, bem como quaisquer outros resultados intermediários significativos, conforme o necessário. Questões para se perguntar ao escrever esta seção:
- _Uma solução inicial foi encontrada e claramente reportada?_
- _O processo de melhoria foi documentado de foma clara, bem como as técnicas utilizadas?_
- _As soluções intermediárias e finais foram reportadas claramente, conforme o processo foi sendo melhorado?_


## IV. Resultados
_(aprox. 2-3 páginas)_

### Modelo de avaliação e validação
Nesta seção, o modelo final e quaisquer qualidades que o sustentem devem ser avaliadas em detalhe. Deve ficar claro como o modelo final foi obtido e por que tal modelo foi escolhido. Além disso, algum tipo de análise deve ser realizada para validar a robustez do modelo e sua solução, como, por exemplo, manipular os dados de entrada ou o ambiente para ver como a solução do modelo é afetada (técnica chamada de análise sensitiva). Questões para se perguntar ao escrever esta seção:
- _O modelo final é razoável e alinhado com as expectativas de solução? Os parâmetros finais do modelo são apropriados?_
- _O modelo final foi testado com várias entradas para avaliar se o modelo generaliza bem com dados não vistos?_
-_O modelo é robusto o suficiente para o problema? Pequenas perturbações (mudanças) nos dados de treinamento ou no espaço de entrada afetam os resultados de forma considerável?_
- _Os resultados obtidos do modelo são confiáveis?_

### Justificativa
Nesta seção, a solução final do seu modelo e os resultados dela obtidos devem ser comparados aos valores de referência (benchmark) que você estabeleceu anteriormente no projeto, usando algum tipo de análise estatística. Você deverá também justificar se esses resultados e a solução são significativas o suficiente para ter resolvido o problema apresentado no projeto. Questões para se perguntar ao escrever esta seção:
- _Os resultados finais encontrados são mais fortes do que a referência reportada anteriormente?_
- _Você analisou e discutiu totalmente a solução final?_
- _A solução final é significativa o suficiente para ter resolvido o problema?_


## V. Conclusão
_(aprox. 1-2 páginas)_

### Foma livre de visualização
Nesta seção, você deverá fornecer alguma forma de visualização que enfatize uma qualidade importante do projeto. A visualização é de forma livre, mas deve sustentar de forma razoável um resultado ou característica relevante sobre o problema que você quer discutir. Questões para se perguntar ao escrever esta seção:
- _Você visualizou uma qualidade importante ou relevante acerca do problema, conjunto de dados, dados de entrada, ou resultados?_
- _A visualização foi completamente analisada e discutida?_
- _Se um gráfico foi fornecido, os eixos, títulos e dados foram claramente definidos?_

### Reflexão

O projeto como um todo foi uma experiência tanto traumática quanto edificadora para mim.

Mergulhar em um problema fora do meu domínio, codificar em Python, aprender sobre como implementar IA como serviço, cloud computing e gpu computing foram áreas que fugiam totalmente de meus conhecimentos prévios, o que me agregou muito conhecimento.

Como um resumo geral de minha jornada na entrega deste projeto, posso afirmar que houveram vários altos e baixos, como também vários imprevistos que impactaram de forma negativa no projeto. A jornada do projeto se resume nos parágrafos abaixo, onde em cada parágrafo aprendi uma lição sobre todo este processo:

Iniciei o projeto utilizando o conjunto de dados escolhidos na proposta de conclusão de projeto que já havia sido aprovada, e percebi que os dados se encontravam muito sujos, com várias imagens de objetos que nada tinham a ver com o problema proposto, além de haver poucas amostras (80 mil ao todo). Para contornar este problema, encontrei outro conjunto de dados, com mais imagens e categorias para classificar. _Aprendi que soluções que demandam dados para ser construídas precisam de **dados de qualidade**_.

Iniciei o desenvolvimento localmente, executando os códigos em minha máquina, logo percebi que os programas estavam demorando absurdamente para mostrar resultados, como uma solução paliativa, migrei o código para o Google Colab, resultando em alguns ajustes no código. Após alguns dias desenvolvendo no Colab, tive novas dificuldades, e novamente tive que migrar meu código para outra plataforma, o [Paperspace](https://www.paperspace.com/). _Aprendi com essas adversidades que o Deep Learning é um método de aprendizado que exige um altíssimo poder computacional, com resultados muito demorados_.

Tive muitas dificuldades em lidar com o Tensorflow, ele apresentou erros diferentes para cada um dos ambientes em que testei meus programas, o que atrasou muito a entrega dos artefatos do projeto. _Aprendi que tenho que migrar para o Pytorch, RÁPIDO_.

A solução desenvolvida ainda está longe do esperado para uma aplicação de produção, mas tenho certeza que foi um passo na direção certa, após as melhorias a ser propostas, tenho certeza que será possível alcançar resultados satisfatórios.

### Melhorias

Como melhorias a este projeto, considero que há a necessidade de estudar técnicas mais avançadas de treinamento de redes neurais, que no momento não podem ser implementadas com o meu conhecimento neste campo de estudo. Estou começando a assistir algumas aulas do curso do [fast.ai](http://www.fast.ai/), e percebi que existem técnicas avançadas para se obter melhores resultados de classificação em redes neurais.

Outro ponto a ser melhorado no projeto consiste na utilização do serviço de uma forma mais completa. Devido ao prazo do projeto chegar ao fim, realizei a entrega do serviço em uma aplicação web simples, o que não agrega o valor necessário para o usuário devido a fatores simples como: entrega de pouca informação(tipo da roupa apenas), falta de persistência da informação recebida e disponibilização de dados analíticos. Como proposta de melhoria para o projeto, seria ideal realizar as seguintes modificações:

* Incrementar as informações geradas pelo serviço, como cor dominante, corte da roupa, tipo de tecido e estilo da roupa(o conjunto de dados de entrada já tem essas informações marcadas nas fotos, seria necessário apenas o treinamento de novos modelos)

* Entregar o serviço em uma plataforma embarcada, para assim fornecer na plataforma os serviços faltantes como persistência e relatórios gerenciais. Era planejado entregar o serviço em um app mobile, mas devido a aproximação do final do prazo e minha falta de experiência em desenvolvimento, tive que abandonar a idéia e entregar na web, mas a aplicação funciona em partes e se encontra em: https://github.com/leandrohmvieira/SmartDrobe

* Um ponto que seria de grande melhora, seria o enquadramento do objeto antes da classificação. Encontrei algumas fontes na internet como o [Detectron](https://github.com/facebookresearch/Detectron) e o [YOLO](https://github.com/zhreshold/mxnet-yolo) que conseguem detectar objetos em uma imagem com várias coisas, isso permitiria cortar a foto antes da classificação, assim teríamos menos ruído durante a classificação das roupas.

* E por fim, seria uma melhora a multiclassificação de uma imagem, onde fosse possível classificar várias roupas simultâneamente, dado uma foto. Realizei algumas pesquisas na internet e vi que isso é possível, porém o material dado pelo curso não abrange este tópico.

-----------

**Antes de enviar, pergunte-se. . .**

- _O relatório de projeto que você escreveu segue uma estrutura bem organizada, similar ao modelo do projeto?_
- Cada seção (particularmente **Análise** e **Metodologia**) foi escrita de maneira clara, concisa e específica? Existe algum termo ou frase ambígua que precise de esclarecimento?
- O público-alvo do seu projeto será capaz de entender suas análises, métodos e resultados?
- Você revisou seu relatório de projeto adequadamente, de forma a minimizar a quantidade de erros gramaticais e ortográficos?
- Todos os recursos usados neste projeto foram corretamente citados e referenciados?
- O código que implementa sua solução está legível e comentado adequadamente?
- O código é executado sem erros e produz resultados similares àqueles reportados?
