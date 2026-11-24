"""Glossário do relatório: o que cada variável significa e como foi calculada (com os parâmetros usados)."""
from __future__ import annotations

import html

import nucleo as nc


def _n(x) -> str:
    return f'{x:g}'.replace('.', ',')


def linhas(p: nc.Parametros, pe) -> list[tuple[str, str, str, str]]:
    """[(grupo, variável, o que significa, como foi calculada)]."""
    uf, ue = p.unidade_forca_saida, p.unidade_emg
    notch = (f'notch IIR de {_n(p.f_notch)} Hz (Q = {_n(p.q_notch)}) + ' if p.usar_notch else '')
    janelas = ', '.join(f'{w}' for w in p.janelas_rfd_ms)
    L = []

    def g(grupo, var, sig, calc):
        L.append((grupo, var, sig, calc))

    # ---------------- pré-processamento
    g('Pré-processamento', 'Força filtrada e tarada',
      'Sinal de cada célula de carga sem ruído de alta frequência e com o zero no repouso.',
      f'Passa-baixa Butterworth de {_n(p.fc_forca)} Hz (ordem {p.ordem_forca}, fase zero). Tara = média do trecho de '
      f'{_n(p.janela_base_s)} s com menor força média do registro (repouso), subtraída de todo o sinal.')
    g('Pré-processamento', 'EMG filtrado',
      'Sinal eletromiográfico sem offset, sem interferência da rede e restrito à banda fisiológica.',
      f'Remoção da média; {notch}passa-banda Butterworth de {_n(p.banda_emg[0])}–{_n(p.banda_emg[1])} Hz (ordem '
      f'{p.ordem_emg}), em cascata e com fase zero (filtragem para frente e para trás).')
    # ---------------- mecânica
    g('Mecânica', f'Pico de força ({uf})', 'Maior força produzida pelo lado durante a CIVM.',
      'Valor máximo da força tarada no trecho analisado.')
    g('Mecânica', 'Janela de CIVM',
      'Trecho de força máxima sustentada, usado para a força média e para o RMS do EMG.',
      f'Janela de {_n(p.janela_civm_s)} s com a maior média da SOMA das células de carga (automática) ou iniciando no '
      'instante definido manualmente; a mesma janela vale para os dois lados e para todos os músculos.')
    g('Mecânica', f'Força média na janela ({uf})', 'Força sustentada na CIVM.',
      'Média da força tarada de cada lado dentro da janela de CIVM.')
    g('Mecânica', 'Onset (s)', 'Instante em que a contração começa.',
      f'Limiar = maior valor entre {p.onset_pct_pico:.0%} do pico e {_n(p.n_sd_ruido)} desvios-padrão do ruído de '
      'repouso; onset = 1 amostra após a última vez em que a força esteve abaixo do limiar antes do pico.')
    g('Mecânica', 'Instante do pico (s) e tempo onset–pico (ms)', 'Quando o pico ocorre e quanto tempo leva desde o onset.',
      'Instante da amostra de força máxima; tempo onset–pico = (pico − onset) em milissegundos.')
    # ---------------- TDF
    g('TDF', f'TDF 0–X ms ({uf}/s)  [= RFD]',
      'Taxa de desenvolvimento de força: rapidez com que a força sobe no início da contração (capacidade "explosiva").',
      f'[F(onset + X) − F(onset)] ÷ X, para X = {janelas} ms.')
    g('TDF', 'TDF normalizada 0–X ms (%Fpico/s)',
      'TDF relativa à força máxima do próprio lado; separa rapidez de força máxima (um lado mais forte tende a ter '
      'TDF absoluta maior).', 'TDF ÷ pico de força × 100.')
    g('TDF', f'Força em X ms ({uf})', 'Quanto de força foi produzido até X ms após o onset.', 'F(onset + X) − F(onset).')
    g('TDF', f'Impulso 0–X ms ({uf}·s)', 'Quantidade de força acumulada no tempo no início da contração.',
      'Integral (regra do trapézio) de [F(t) − F(onset)] de onset até onset + X.')
    g('TDF', f'TDF máxima 0–200 ms ({uf}/s) e seu instante',
      'Maior velocidade de aumento da força no início da contração.',
      f'Maior inclinação ΔF/Δt em janelas deslizantes de {_n(pe.janela_tdf_max_ms)} ms dentro dos primeiros 200 ms após o '
      'onset; o instante é medido a partir do onset.')
    # ---------------- EMG
    g('Eletromiografia', f'RMS ({ue})', 'Amplitude do EMG: nível de ativação elétrica do músculo na CIVM.',
      'Raiz da média dos quadrados do EMG filtrado dentro da janela de CIVM.')
    g('Eletromiografia', f'Eficiência neuromuscular — ENM ({uf}/{ue})',
      'Força produzida por unidade de ativação elétrica. No agonista, maior ENM = mais força para a mesma ativação. '
      'No antagonista não representa eficiência daquele músculo.',
      'Força média na janela de CIVM do mesmo lado ÷ RMS do músculo.')
    g('Eletromiografia', 'Auditoria espectral — atenuação (dB)',
      'Confirma que a interferência da rede elétrica foi removida.',
      f'10·log10(potência do sinal bruto ÷ potência do filtrado) em {_n(p.f_notch)} ± 2 Hz, com espectros de Welch.')
    # ---------------- cocontração
    g('Cocontração', 'Agonista e antagonista',
      f'Na tarefa "{pe.tarefa}", o agonista ({pe.agonista}) produz o movimento e o antagonista ({pe.antagonista}) se '
      'opõe a ele. A cocontração é a ativação simultânea dos dois; ela aumenta a rigidez e a estabilidade da articulação, '
      'mas reduz a força líquida medida.', 'Analisada separadamente em cada lado (bíceps D × tríceps D; bíceps E × tríceps E).')
    g('Cocontração', 'Envoltória RMS', 'Curva suave da amplitude do EMG ao longo do tempo.',
      f'RMS móvel centrado de {_n(pe.envoltoria_ms)} ms sobre o EMG filtrado. Se a RMS de referência (CIVM própria) de '
      'cada músculo for informada, o EMG é antes dividido por ela (% da CIVM).')
    g('Cocontração', 'Razão antagonista/agonista (%)',
      'Ativação do antagonista em proporção à do agonista (índice de coativação).',
      'RMS do antagonista ÷ RMS do agonista × 100, em cada janela (janela de CIVM e 0–50/0–100/0–200 ms após o '
      'onset da força do mesmo lado).')
    g('Cocontração', 'Índice de cocontração de Falconer & Winter (%)',
      'Proporção do tempo-amplitude em que os dois músculos estão ativos ao mesmo tempo: 0% = sem cocontração; 100% = '
      'ativação idêntica dos dois.', '2 × Σ mín(envoltória agonista, envoltória antagonista) ÷ Σ (agonista + antagonista) '
      '× 100, somando as amostras da janela (Falconer & Winter, 1985).')
    g('Cocontração', 'Normalização', 'Sem normalização, os índices dependem do ganho e da posição de cada eletrodo; '
      'comparações entre músculos, lados e dias ficam limitadas.',
      'Recomendado informar a RMS de CIVM de cada músculo na sua ação de agonista (ex.: tríceps numa CIVM de extensão).')
    # ---------------- assimetria
    g('Assimetria', 'Assimetria D–E (%)', 'Diferença entre os lados relativa ao lado maior; sinal positivo = direito maior.',
      '100 × (D − E) ÷ máx(|D|, |E|).')
    g('Assimetria', 'Índice de simetria (%)', 'Diferença entre os lados relativa à média dos dois.',
      '100 × (D − E) ÷ [(D + E) / 2].')
    g('Assimetria', 'LSI — índice de simetria entre membros (%)', 'Quanto o lado menor representa do maior (100% = simétrico).',
      '100 × menor ÷ maior.')
    g('Assimetria', 'Sinalização', 'Destaca as variáveis com assimetria relevante.',
      f'|Assimetria| > {_n(pe.limiar_assimetria_pct)}% (limiar ajustável).')
    g('Assimetria', 'Comparação bilateral por músculo', 'RMS e ENM de cada músculo nos dois lados.',
      'Mesma fórmula de assimetria aplicada ao RMS e à ENM do músculo direito e do esquerdo.')
    # ---------------- qualidade
    g('Qualidade', 'Avisos automáticos', 'Problemas que podem invalidar os números.',
      'Saturação do EMG (amostras no valor extremo), pico fora da janela de CIVM, força já acima do limiar no início, '
      'janela de TDF que ultrapassa o pico ou o fim do registro, irregularidades na coluna de tempo, canais duplicados.')
    return L


def html_glossario(p: nc.Parametros, pe) -> str:
    corpo, grupo_atual = [], None
    for grupo, var, sig, calc in linhas(p, pe):
        if grupo != grupo_atual:
            corpo.append(f'<tr><td colspan="3" style="background:#eaf0f7;font-weight:bold;text-align:left">'
                         f'{html.escape(grupo)}</td></tr>')
            grupo_atual = grupo
        corpo.append(f'<tr><td style="text-align:left;width:24%"><b>{html.escape(var)}</b></td>'
                     f'<td style="text-align:left;width:34%">{html.escape(sig)}</td>'
                     f'<td style="text-align:left">{html.escape(calc)}</td></tr>')
    return ('<table><tr><th>Variável</th><th>O que significa</th><th>Como foi calculada</th></tr>'
            + ''.join(corpo) + '</table>')
