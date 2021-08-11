# indoors-trajectory-prediction

Neste repositório estão presentes três modelagens para o problema de interpolação de trajetórias em ambientes internos:
  - Random Forest
  - MLP (um tipo de Rede Neural Artificial)
  - interpolação linear (base line) 

## Modelagem do Random Forest e da interpolação linear

O algoritmo Random Forest foi modelado utilizando o RandomForestRegressor em conjunto ao MultiOutputRegressor, ambos componentes do módulo [scikit-learn](https://scikit-learn.org/stable/install.html), usando apenas a parametrização default. O algoritmo foi executado dez vezes, devido a sua característica estocástica, e o dado foi dividido de forma que 80% fosse usado para treino, e 20% para teste. O código referente a este modelo está presente em [modelo_rf_int.py](modelo_rf_int.py). Neste arquivo também está presente o código da interpolação linear.
 
## Modelagem da RNA

Utilizou-se a implementação do [tensorflow](https://www.tensorflow.org/install?hl=pt-br) para realizar todos os testes deste artigo. A rede neural construída possui quatro camadas, são elas da primeira à ultima:
  1. 2000 neurônios utilizando ReLU
  2. 500 neurônios utilizando ReLU
  3. 20 neurônios utilizando ReLU
  4. 2 neurônios utilizando tangente hiperbólica

Por fim, o otimizador foi o Adam e o learning rate inicial de 0.0001 (são utilizados callbacks para reduzir este valor).

Similarmente ao que foi feito no modelo anterior, foram realizadas dez execuções para todos os testes, dividindo os dados em 64\% para treino, 20\% para validação e 16\% para teste. O código referente a este modelo está presente em [modelo_rna.py](modelo_rna.py)

## Organização dos arquivos

Os datasets utilizados estão na pasta [datasets](datasets). Nela, os arquivos, [class.txt](datasets/class.txt), [lab.txt](datasets/lab.txt), [market.txt](datasets/market.txt) e [real.csv](datasets/real.csv) são correspondentes respectivamente aos ambientes citados no artigo Ambiente 1, Ambiente 2, Ambiente 3 e Ambiente 4.

Durante a execução dos códigos citados anteriormente, são gerados arquivos para cada métrica em cada valor de d. Nestes arquivos, cada linha corresponde a uma execução e os valores correspondem às métricas obtidas para x e y em cada uma destas execuções. Estes arquivos são gravados dentro de uma pasta raíz [validacao](validacao). Dentro desta pasta, os arquivos são organizados de acordo com o dataset e modelo.
