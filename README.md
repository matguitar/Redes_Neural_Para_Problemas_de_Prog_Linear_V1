# [cite_start]Rede Neural para Aproximação de Soluções em Problemas de Otimização Linear [cite: 1]

## 1. Problema
[cite_start]Problemas de **Programação Linear (PL)** são onipresentes em setores como logística, produção, telecomunicações e finanças[cite: 3]. [cite_start]Embora os métodos exatos (como o Simplex) sejam a referência de precisão, eles apresentam um custo de execução por instância[cite: 4].

[cite_start]Este projeto propõe uma **rede neural** para aprender a aproximar a solução ótima de problemas de PL de pequeno porte[cite: 5].

---

## 2. Metodologia

### 2.1 Fonte dos Dados
* [cite_start]Os dados utilizados são gerados artificialmente (sintéticos)[cite: 7].
* [cite_start]Cada instância consiste em um problema de programação linear com **duas variáveis** e **duas restrições**[cite: 7].
* [cite_start]As soluções ótimas são obtidas via biblioteca **PuLP**, servindo como base de comparação para o treinamento[cite: 8].

### 2.2 Objetivo
* [cite_start]Treinar uma rede neural para receber os coeficientes do problema como entrada e prever a solução ótima aproximada como saída[cite: 10].
* [cite_start]Atuar como um **"solucionador rápido"**, ideal para aplicações que exigem respostas em tempo real[cite: 11].

### [cite_start]2.3 Descrição Matemática [cite: 13]
Dado o vetor de entrada $[c_{1}, c_{2}, a_{1}, a_{2}, a_{3}, a_{4}, b_{1}, b_{2}]$, o modelo busca prever a solução $[x_{1}^{*}, x_{2}^{*}]$ para:

**Maximizar:**
[cite_start]$$c_{1}x_{1} + c_{2}x_{2}$$ [cite: 14]

**Sujeito a:**
[cite_start]$$a_{1}x_{1} + a_{2}x_{2} \le b_{1}$$ [cite: 15]
[cite_start]$$a_{3}x_{1} + a_{4}x_{2} \le b_{2}$$ [cite: 16]
[cite_start]$$x_{1}, x_{2} \ge 0$$ [cite: 16]

---

## [cite_start]3. Resolução do Problema [cite: 18]

| Etapa | Descrição |
| :--- | :--- |
| **1. Geração de Dados** | Gerar $N$ problemas de PL aleatórios e resolvê-los com a biblioteca PuLP. |
| **2. Preparação de Dados** | Converter os coeficientes e soluções em tensores para processamento. |
| **3. Treinamento da Rede** | Treinar o modelo utilizando o framework **PyTorch**. |

### [cite_start]Configuração da Rede Neural[cite: 18]:
* **Arquitetura:** Rede $8 \rightarrow 32 \rightarrow 16 \rightarrow 2$
* **Função de Ativação:** ReLU
* **Otimizador:** Adam
* **Função de Perda:** MSE (Erro Quadrático Médio)
* **Framework:** PyTorch

---

## 4. Limitações
* [cite_start]A rede é treinada estritamente com dados sintéticos[cite: 20].
* [cite_start]O modelo é limitado a problemas com duas variáveis de decisão e duas restrições, não sendo capaz de generalizar para problemas maiores[cite: 20].

---
