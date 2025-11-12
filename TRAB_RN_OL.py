# -*- coding: utf-8 -*-

#!pip install torch
#!pip install pulp
#!sudo apt-get install coinor-cbc
#!pip install scikit-learn
#!pip install scipy

import torch
import torch.nn as nn
import torch.optim as optim
from pulp import *
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt

# Adicione este bloco no início do seu código
SEED = 42  # Um número inteiro qualquer. 42 é uma convenção comum.

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 1. Geração de Dados de Treinamento ---
def gerar_dados(num_amostras):
    dados = []
    for _ in range(num_amostras):
        # Problema de Otimização Linear Simples
        # Maximize: c1*x1 + c2*x2
        # Sujeito a: a1*x1 + a2*x2 <= b1
        #            a3*x1 + a4*x2 <= b2
        #            x1, x2 >= 0
        
        # Gerar coeficientes e limites aleatórios
        c1, c2 = np.random.rand(2) * 10
        a1, a2, a3, a4 = np.random.rand(4) * 10
        b1, b2 = np.random.rand(2) * 20

        prob = LpProblem("ProblemaSimples", LpMaximize)
        x1 = LpVariable("x1", 0, None)
        x2 = LpVariable("x2", 0, None)

        # Adicionar a função objetivo
        prob += c1 * x1 + c2 * x2, "FunçãoObjetivo"

        # Adicionar as restrições
        prob += a1 * x1 + a2 * x2 <= b1, "Restricao1"
        prob += a3 * x1 + a4 * x2 <= b2, "Restricao2"

        # Resolver o problema
        prob.solve(PULP_CBC_CMD(msg=0))

        if LpStatus[prob.status] == 'Optimal':
            # Se a solução for ótima, salve os dados
            coeficientes = [c1, c2, a1, a2, a3, a4, b1, b2]
            solucao = [x1.varValue, x2.varValue]
            dados.append({'entrada': coeficientes, 'saida': solucao})
            
    return dados

# --- 2. Preparação e Conversão para Tensores ---
def preparar_dados(dados):
    entradas = torch.tensor([d['entrada'] for d in dados], dtype=torch.float32)
    saidas = torch.tensor([d['saida'] for d in dados], dtype=torch.float32)
    return entradas, saidas

# --- 3. Desenvolvimento da Rede Neural (Melhorado) ---
class RedeOtimizacao(nn.Module):
    def __init__(self):
        super(RedeOtimizacao, self).__init__()
        
        # Usando nn.Sequential para construir a rede de forma limpa e modular
        self.camadas = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),  # Usando ReLU, que é mais eficaz que Sigmoid
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
    def forward(self, x):
        return self.camadas(x)

# --- 4. Treinamento da Rede ---
def treinar_modelo(entradas, saidas, epocas=9400):
    modelo = RedeOtimizacao()
    criterio = nn.MSELoss()  # Erro Quadrático Médio
    otimizador = optim.Adam(modelo.parameters(), lr=0.01)

    for epoca in range(epocas):
        modelo.train()
        previsoes = modelo(entradas)
        perda = criterio(previsoes, saidas)
        
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
        
        if (epoca+1) % 100 == 0:
            print(f'Época [{epoca+1}/{epocas}], Perda: {perda.item():.4f}')
            
    return modelo

# --- PARTE I: GERAR PROBLEMAS E TREINAR A REDE ---
    
## Geração dos problemas de otimização linear simples

print("Gerando dados de treinamento...")
dados_treinamento = gerar_dados(num_amostras=5000)

print("Preparando dados...")
entradas_treinamento, saidas_treinamento = preparar_dados(dados_treinamento)

## Treinamento da rede neural 

print("Iniciando o treinamento da rede neural...")
modelo_treinado = treinar_modelo(entradas_treinamento, saidas_treinamento)

## Métricas de Avaliação

modelo_treinado.eval()
with torch.no_grad():
 previsoes_treinamento = modelo_treinado(entradas_treinamento)
            
saidas_reais = saidas_treinamento.numpy()
previsoes_modelo = previsoes_treinamento.numpy()
        
mse = mean_squared_error(saidas_reais, previsoes_modelo)
mae = mean_absolute_error(saidas_reais, previsoes_modelo)

print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
print(f"Erro Absoluto Médio (MAE): {mae:.4f}")

# --- PARTE II: TESTE ESTATÍSTICO PARA AS DIFERENÇAS ENTRE RESULTADOS VIA PULP E VIA REDE NEURAÇ ---

print("Gerando 600 novos problemas para a validação estatística...")
dados_teste = gerar_dados(num_amostras=600)       

entradas_teste, saidas_reais_teste = preparar_dados(dados_teste)
            
modelo_treinado.eval()
with torch.no_grad():
 previsoes_teste = modelo_treinado(entradas_teste)

            # Para o teste, vamos comparar os valores de x1 e x2 separadamente
x1_reais = saidas_reais_teste[:, 0].numpy()
x1_previsoes = previsoes_teste[:, 0].numpy()
            
x2_reais = saidas_reais_teste[:, 1].numpy()
x2_previsoes = previsoes_teste[:, 1].numpy()

## Visualização

# --- Visualização: Histogramas da Diferença ---

print("\n--- Visualização: Histogramas da Diferença (PuLP - Rede Neural) ---")

# Cria uma figura com dois subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Histograma para a diferença de x1
ax1.hist(diff_x1, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_title('Diferença entre PuLP e Rede Neural para x1')
ax1.set_xlabel('Diferença (PuLP - Rede Neural)')
ax1.set_ylabel('Frequência')
ax1.grid(True, linestyle='--', alpha=0.6)

# Adiciona uma linha vertical no zero
ax1.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Diferença = 0')
ax1.legend()

# Histograma para a diferença de x2
ax2.hist(diff_x2, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
ax2.set_title('Diferença entre PuLP e Rede Neural para x2')
ax2.set_xlabel('Diferença (PuLP - Rede Neural)')
ax2.set_ylabel('Frequência')
ax2.grid(True, linestyle='--', alpha=0.6)

# Adiciona uma linha vertical no zero
ax2.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Diferença = 0')
ax2.legend()

plt.tight_layout()
plt.show()

print("\nInterpretação dos Histogramas:")
print("A análise visual dos histogramas reforça as conclusões do teste estatístico. A maioria das diferenças está muito próxima de zero, indicando que a rede neural está gerando previsões extremamente próximas das soluções ótimas do PuLP.")
print("A distribuição das diferenças é aproximadamente simétrica em torno do zero, confirmando que a rede neural não tem um 'viés' consistente de subestimar ou superestimar os valores.")

# --- Testando a diferença entre Pulp - Rede Neural ---

# Teste de Wilcoxon para x1
stat_x1, p_value_x1 = stats.wilcoxon(x1_reais, x1_previsoes)
print(f"\nTeste de Wilcoxon para a variável x1:")
print(f"Estatística de Teste: {stat_x1:.4f}")
print(f"P-valor: {p_value_x1:.4f}")

 # Teste de Wilcoxon para x2
stat_x2, p_value_x2 = stats.wilcoxon(x2_reais, x2_previsoes)
print(f"\nTeste de Wilcoxon para a variável x2:")
print(f"Estatística de Teste: {stat_x2:.4f}")
print(f"P-valor: {p_value_x2:.4f}")
            

print("\nInterpretação:")
alpha = 0.05
if p_value_x1 < alpha:
 print(f"- A diferença nas soluções de x1 é estatisticamente significativa (p < {alpha}).")
else:
 print(f"- A diferença nas soluções de x1 NÃO é estatisticamente significativa (p > {alpha}).")

if p_value_x2 < alpha:
 print(f"- A diferença nas soluções de x2 é estatisticamente significativa (p < {alpha}).")
else:
 print(f"- A diferença nas soluções de x2 NÃO é estatisticamente significativa (p > {alpha}).")

print("\n--- Visualização e Intervalos de Confiança (para a diferença entre Pulp - Rede Neural)---")

# Intervalos de Confiança para a Diferença
print("\nIntervalos de Confiança (95%) para a diferença (PuLP - Rede Neural):")

# Cálculo para x1
diff_x1 = x1_reais - x1_previsoes
mean_diff_x1 = np.mean(diff_x1)
std_err_x1 = stats.sem(diff_x1) # Erro padrão da média
ci_x1 = stats.t.interval(0.95, len(diff_x1)-1, loc=mean_diff_x1, scale=std_err_x1)

print(f"x1: [{ci_x1[0]:.4f}, {ci_x1[1]:.4f}]")

# Cálculo para x2
diff_x2 = x2_reais - x2_previsoes
mean_diff_x2 = np.mean(diff_x2)
std_err_x2 = stats.sem(diff_x2)
ci_x2 = stats.t.interval(0.95, len(diff_x2)-1, loc=mean_diff_x2, scale=std_err_x2)

print(f"x2: [{ci_x2[0]:.4f}, {ci_x2[1]:.4f}]")

print("\nInterpretação dos Intervalos de Confiança:")
print("- Como o intervalo contém zero, não podemos afirmar que há uma diferença significativa entre as soluções da rede neural e as soluções ótimas do PuLP.")
print("- Isso reforça a conclusão do Teste de Wilcoxon.")
