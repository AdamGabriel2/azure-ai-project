# azure-ai-project

# Projeto de Machine Learning no Azure

Este projeto prático utiliza o Azure AI Services para explorar capacidades de Machine Learning, incluindo a criação de um modelo de Content Safety e o uso do Automated Machine Learning no Azure Machine Learning para treinar e implantar um modelo preditivo de aluguel de bicicletas.

## Azure Automated Machine Learning

### Visão Geral

Nesta parte do projeto, utilizamos o Automated Machine Learning no Azure Machine Learning para criar, treinar, avaliar e implantar um modelo preditivo. O modelo prevê o número de aluguéis de bicicletas com base em características sazonais e meteorológicas.

### Estrutura do Projeto

```
.
└── train.py
└── evaluate.py
└── deploy.py
└── README.md
```

### Scripts

- **train.py**: Script para treinar o modelo usando Automated Machine Learning.
- **evaluate.py**: Script para avaliar o modelo treinado.
- **deploy.py**: Script para implantar o modelo como um serviço web.

### Como Executar

1. Clone este repositório.
2. Instale as dependências necessárias: `pip install -r requirements.txt`.
3. Execute os scripts na ordem adequada.

### Personalização

Certifique-se de personalizar os scripts conforme necessário, ajustando parâmetros, definindo a arquitetura do modelo, e lidando com seu próprio conjunto de dados.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para melhorar este projeto.
