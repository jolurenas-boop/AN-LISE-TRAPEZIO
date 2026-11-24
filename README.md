# Trapézio — CIVM e fadiga a 50% da CIVM

App Streamlit para **1 eletrodo de EMG no trapézio + 1 célula de carga**.

## Arquivos (um por teste)
| Col. | Conteúdo |
|---|---|
| 1 | Tempo |
| 2 | Trapézio (EMG) |
| 3 | Célula de carga |

As colunas, o separador e o cabeçalho podem ser alterados na barra lateral. O app confere se a coluna de força tem
sinal de força e a de EMG sinal de EMG, e não mostra resultados se o mapeamento estiver errado.

## O que é calculado
- **CIVM:** tara, onset, pico, janela de CIVM (1 s), TDF 0–50/100/200 ms (absoluta e normalizada), TDF máxima,
  impulso, RMS, ENM, auditoria espectral (notch).
- **Fadiga 50%:** força em % da CIVM e % do tempo no alvo (±5%), início do patamar (média ± 2 DP por ≥ 2 s, com
  verificação de CV e deriva; ajuste manual possível), janelas de 0,5 s com 50% de sobreposição, MDF/MNF/RMS/ENM por
  janela, inclinação com p corrigido para a sobreposição, queda da MDF (extremos e regressão), exclusão de artefatos.
- **Integração CIVM → fadiga:** RMS da fadiga em % da RMS da CIVM, ENM em % da ENM da CIVM. Se o arquivo da fadiga
  não tiver repouso inicial, a tara da célula é emprestada da CIVM.
- **Relatório PDF:** identificação, CIVM, fadiga, controle de qualidade, métodos, glossário e resumo visual.

A fadiga é opcional: só com o arquivo da CIVM, o app faz a análise da CIVM.

## Executar
```
pip install -r requirements.txt
streamlit run app.py
```
