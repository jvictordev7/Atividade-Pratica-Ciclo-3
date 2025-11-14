# 🚀 Atividade Prática – Ciclo 3

Projeto em Python que implementa do zero um Perceptron para classificar flores do dataset Iris (Setosa vs Versicolor) e visualizar a fronteira de decisão em 2D.

## 💡 Visão geral
- Carrega o dataset Iris diretamente do `scikit-learn`.
- Mantém apenas as classes Setosa (0) e Versicolor (1) e usa as duas medidas da pétala (comprimento e largura) para facilitar o plot em duas dimensões.
- Treina o Perceptron personalizado, avalia no conjunto de teste e exibe métricas básicas.
- Gera um gráfico com as regiões de decisão e os pontos de treino/teste.

## 📦 Conteúdo
- `perceptron_iris.py` – script principal com a implementação do Perceptron, treinamento, avaliação e visualização.
- `README.md` – guia rápido de configuração, execução e entrega.

## ⚙️ Requisitos
Certifique-se de estar dentro de um ambiente virtual e instale as dependências:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## 🚀 Como executar
1. Criar ambiente virtual: `python -m venv .venv`
2. Ativar (PowerShell): `.\.venv\Scripts\activate`
3. Instalar dependências: `pip install numpy pandas matplotlib scikit-learn`
4. Rodar o script: `python perceptron_iris.py`

## 📊 O que você verá
- No terminal:
  - Classes utilizadas
  - Pesos finais e bias do Perceptron
  - Acurácia no conjunto de teste
- Na tela: gráfico com a fronteira de decisão, áreas das classes e pontos de treino/teste.

Se precisar salvar a figura sem abrir a janela, defina `MPLBACKEND=Agg` antes de executar o script.
