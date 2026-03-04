# Rede Neural para Aproximação de Soluções em Problemas de Otimização Linear (Protótipo)

## 1. Problema
Problemas de **Programação Linear (PL)** são onipresentes em setores como logística, produção, telecomunicações e finanças. Embora os métodos exatos (como o Simplex) sejam a referência de precisão, eles apresentam um custo de execução por instância.

Este projeto propõe uma rede neural para aprender a aproximar a solução ótima de problemas de PL de pequeno porte.

---

## 2. Metodologia

### 2.1 Fonte dos Dados
* Os dados utilizados são sintéticos.
* Cada instância consiste em um problema de programação linear com **duas variáveis** e **duas restrições**.
* As soluções ótimas são obtidas via biblioteca **PuLP**, servindo como base de comparação para o treinamento.

### 2.2 Objetivo
* Treinar uma rede neural para receber os coeficientes do problema como entrada e prever a solução ótima aproximada como saída.
* Atuar como um solucionador rápido, ideal para aplicações que exigem respostas em tempo real.

### 2.3 Descrição Matemática
Dado o vetor de entrada $[c_{1}, c_{2}, a_{1}, a_{2}, a_{3}, a_{4}, b_{1}, b_{2}]$, o modelo busca prever a solução $[x_{1}^{\*}, x_{2}^{\*}]$ para:

**Maximizar:**
$$c_{1}x_{1} + c_{2}x_{2}$$

**Sujeito a:**
$$a_{1}x_{1} + a_{2}x_{2} \le b_{1}$$
$$a_{3}x_{1} + a_{4}x_{2} \le b_{2}$$
$$x_{1}, x_{2} \ge 0$$

---

## 3. Resolução do Problema

| Etapa | Descrição |
| :--- | :--- |
| **1. Geração de Dados** | Gerar $N$ problemas de PL aleatórios e resolvê-los com a biblioteca PuLP. |
| **2. Preparação de Dados** | Converter os coeficientes e soluções em tensores para processamento. |
| **3. Treinamento da Rede** | Treinar o modelo utilizando o framework **PyTorch**. |

### Configuração da Rede Neural:
* **Arquitetura:** Rede $8 \rightarrow 32 \rightarrow 16 \rightarrow 2$
* **Função de Ativação:** ReLU
* **Otimizador:** Adam
* **Função de Perda:** MSE (Erro Quadrático Médio)
* **Framework:** PyTorch

---

## 4. Limitações
* A rede é treinada estritamente com dados sintéticos.
* O modelo é limitado a problemas com duas variáveis de decisão e duas restrições, não sendo capaz de generalizar para problemas maiores.

---
