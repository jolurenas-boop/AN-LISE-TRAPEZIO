"""Trapézio + 1 célula de carga: integra a CIVM e o teste de fadiga a 50% da CIVM.

A CIVM fornece o pico de força (que define o alvo de X% da CIVM) e a RMS máxima do trapézio (que normaliza o EMG
da fadiga em % da CIVM). A fadiga é analisada pelo mesmo núcleo do app de fadiga (patamar, janelas, MDF/MNF/RMS/ENM).
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import avancado as av
import fadiga_individual as fi
import fadiga_nucleo as fn
import glossario as gl
import nucleo as nc
from fadiga_graficos import fig_emg as f_emg_fad, fig_forca as f_forca_fad, fig_tendencia
from graficos import fig_emg as f_emg_civm, fig_forca as f_forca_civm, fig_onset, fig_psd
from relatorio import CSS, _figura, _tabela, _tabela_txt

AZUL, TXT, TXT2, GRADE, CRIT, ALVO = '#2a78d6', '#0b0b0b', '#52514e', '#e6e5e0', '#d03b3b', '#1baf7a'


@dataclass(frozen=True)
class ParametrosTeste:
    alvo_pct: float = 50.0          # alvo da fadiga (% da CIVM)
    tolerancia_pct: float = 5.0     # faixa aceitável: alvo ± tolerância (pontos de % da CIVM)


# ------------------------------------------------------------------ integração CIVM -> fadiga
def integra(r_civm, r_fad, pt: ParametrosTeste) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(tabela de integração, janelas da fadiga com RMS em % da CIVM, avisos)."""
    avisos = []
    f_id = r_civm.forca_ids[0]
    pico = float(r_civm.mec[f_id]['pico'])
    col_rms = [c for c in r_civm.df_emg.columns if c.startswith('RMS')][0]
    rms_civm = float(r_civm.df_emg.iloc[0][col_rms])
    col_enm = [c for c in r_civm.df_emg.columns if c.startswith('Eficiência')][0]
    enm_civm = float(r_civm.df_emg.iloc[0][col_enm])
    fr = next(iter(r_fad.forca.values()))
    alvo = pt.alvo_pct / 100 * pico
    fs = r_fad.parametros['fs']
    fm = fr['fm'][fr['i_ini']:fr['i_fim']]
    pct_serie = 100 * fm / pico
    no_alvo = 100 * float(np.mean(np.abs(pct_serie - pt.alvo_pct) <= pt.tolerancia_pct)) if len(fm) else np.nan
    e = next(iter(r_fad.emg.values()))
    jan = e['janelas'].copy()
    jan['RMS_pct_CIVM'] = 100 * jan['RMS'] / rms_civm if rms_civm > 0 else np.nan
    v = jan[jan['valida']]
    t = v['t_centro_s'].to_numpy() - e['i_ini'] / fs
    reg = fn.regressao(t, v['RMS_pct_CIVM'].to_numpy()) if len(v) >= 3 else {'slope': np.nan}
    ext = r_fad.parametros['extremos_s']
    ini = float(v.loc[t <= t.min() + ext, 'RMS_pct_CIVM'].mean()) if len(v) else np.nan
    fim = float(v.loc[t >= t.max() - ext, 'RMS_pct_CIVM'].mean()) if len(v) else np.nan
    enm_ini, enm_fim = e['resumo']['ENM_ini'], e['resumo']['ENM_fim']
    uf = r_fad.parametros['unidade_forca']
    linhas = {
        f'Pico de força na CIVM ({uf})': pico,
        f'Alvo de {pt.alvo_pct:g}% da CIVM ({uf})': alvo,
        f'Força média no patamar da fadiga ({uf})': fr['media'],
        'Força média no patamar (% da CIVM)': 100 * fr['media'] / pico,
        'Força inicial do patamar (% da CIVM)': 100 * fr['forca_ini'] / pico,
        'Força final do patamar (% da CIVM)': 100 * fr['forca_fim'] / pico,
        f'% do tempo do patamar no alvo ({pt.alvo_pct:g} ± {pt.tolerancia_pct:g}% da CIVM)': no_alvo,
        f'RMS do trapézio na CIVM ({r_civm.unidade_emg})': rms_civm,
        'RMS inicial da fadiga (% da RMS da CIVM)': ini,
        'RMS final da fadiga (% da RMS da CIVM)': fim,
        'Inclinação do RMS normalizado (% da CIVM/s)': reg['slope'],
        f'ENM na CIVM ({uf}/{r_civm.unidade_emg})': enm_civm,
        'ENM inicial da fadiga (% da ENM da CIVM)': 100 * enm_ini / enm_civm if enm_civm else np.nan,
        'ENM final da fadiga (% da ENM da CIVM)': 100 * enm_fim / enm_civm if enm_civm else np.nan,
    }
    pico_rms = np.nanmax([ini, fim]) if np.isfinite([ini, fim]).any() else np.nan
    if np.isfinite(pico_rms) and pico_rms > 110:
        avisos.append(f'O RMS do trapézio na fadiga chega a {pico_rms:.0f}% da RMS da CIVM, acima do máximo: a CIVM pode '
                      f'não ter sido máxima, ou o eletrodo/ganho mudou entre os testes. A normalização pela CIVM fica '
                      f'comprometida.')
    pct = 100 * fr['media'] / pico
    if abs(pct - pt.alvo_pct) > pt.tolerancia_pct:
        avisos.append(f'A força média do patamar ficou em {pct:.1f}% da CIVM, fora do alvo de {pt.alvo_pct:g} ± '
                      f'{pt.tolerancia_pct:g}%. Confira se a CIVM e a fadiga são do mesmo dia/voluntário e se o alvo foi '
                      f'seguido.')
    return pd.DataFrame({'Valor': linhas}), jan, avisos


# ------------------------------------------------------------------ resumo visual
def _eixo(ax, titulo):
    ax.set_title(titulo, fontsize=9, loc='left', color=TXT, fontweight='bold')
    ax.tick_params(labelsize=7.5, colors=TXT2, length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#9a9890')
    ax.grid(axis='y', color=GRADE, lw=0.6)
    ax.set_axisbelow(True)


def _tile(ax, titulo, valor, detalhe, alerta=False):
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='#f4f3ef', zorder=0))
    if alerta:
        ax.add_patch(plt.Rectangle((0, 0), 0.025, 1, transform=ax.transAxes, color=CRIT, zorder=1))
    ax.text(0.07, 0.80, titulo, transform=ax.transAxes, fontsize=7.3, color=TXT2, va='top')
    ax.text(0.07, 0.47, valor, transform=ax.transAxes, fontsize=14, color=TXT, va='center', fontweight='bold')
    ax.text(0.07, 0.13, detalhe, transform=ax.transAxes, fontsize=6.8, color=TXT2, va='center')


def fig_resumo(r_civm, ex, r_fad, integ, jan_norm, pt):
    uf = r_civm.unidade_forca
    fig = plt.figure(figsize=(8.27, 9.6))
    linhas_gs = 5 if r_fad is not None else 3
    gs = GridSpec(linhas_gs, 4, figure=fig, height_ratios=[0.6] + [1.1] * (linhas_gs - 1), hspace=0.72, wspace=0.45,
                  left=0.08, right=0.97, top=0.975, bottom=0.05)
    f_id = r_civm.forca_ids[0]
    m = r_civm.mec[f_id]
    t100 = [i for i in ex.tdf.index if i.startswith('TDF 0–100')]
    col_t = ex.tdf.columns[0]
    _tile(fig.add_subplot(gs[0, 0]), 'Pico de força (CIVM)', f'{nc.fmt(m["pico"], 1)}', uf)
    _tile(fig.add_subplot(gs[0, 1]), 'TDF 0–100 ms (CIVM)', nc.fmt(ex.tdf.loc[t100[0], col_t], 0) if t100 else '—',
          f'{uf}/s')
    if r_fad is not None:
        e = next(iter(r_fad.emg.values()))['resumo']
        pct = integ.loc['Força média no patamar (% da CIVM)', 'Valor']
        fora = abs(pct - pt.alvo_pct) > pt.tolerancia_pct
        _tile(fig.add_subplot(gs[0, 2]), 'Força na fadiga', f'{nc.fmt(pct, 1)}% CIVM',
              f'alvo {pt.alvo_pct:g} ± {pt.tolerancia_pct:g}%' + (' — FORA' if fora else ''), fora)
        _tile(fig.add_subplot(gs[0, 3]), 'Queda da MDF (regressão)', f'{nc.fmt(e["Queda_MDF_reg"], 1)}%',
              f'inclinação {nc.fmt(e["MDF_slope"], 3)} Hz/s')
    else:
        col_rms = [c for c in r_civm.df_emg.columns if c.startswith('RMS')][0]
        _tile(fig.add_subplot(gs[0, 2]), 'RMS do trapézio (CIVM)', nc.fmt(r_civm.df_emg.iloc[0][col_rms], 0),
              r_civm.unidade_emg)
        fig.add_subplot(gs[0, 3]).axis('off')

    # A. CIVM: força com janela
    ax = fig.add_subplot(gs[1, :])
    _eixo(ax, 'A. CIVM — força e janela de 1 s de maior força')
    k = max(1, len(r_civm.t_seg) // 4000)
    ax.plot(r_civm.t_seg[::k], r_civm.forcas[f_id][::k], color=AZUL, lw=1.6)
    ax.axvspan(r_civm.t_seg[r_civm.janela[0]], r_civm.t_seg[r_civm.janela[1] - 1], color=ALVO, alpha=0.15, lw=0)
    ax.axvline(r_civm.t_seg[m['idx_onset']], color=TXT2, ls=':', lw=1)
    ax.text(r_civm.t_seg[m['idx_onset']], ax.get_ylim()[1] * 0.92, ' onset', fontsize=7, color=TXT2)
    ax.set_ylabel(uf, fontsize=7.5, color=TXT2)
    ax.set_xlabel('Tempo (s)', fontsize=7.5, color=TXT2)

    # B. TDF
    ax = fig.add_subplot(gs[2, :2])
    _eixo(ax, 'B. TDF da CIVM')
    lin = [i for i in ex.tdf.index if i.startswith('TDF 0–')]
    vals = [float(ex.tdf.loc[i, col_t]) for i in lin]
    ax.bar(range(len(lin)), vals, color=AZUL, width=0.55)
    for x, v in enumerate(vals):
        ax.text(x, v, nc.fmt(v, 0), ha='center', va='bottom', fontsize=7, color=TXT2)
    ax.set_xticks(range(len(lin)))
    ax.set_xticklabels([i.split(' (')[0].replace('TDF ', '') for i in lin], fontsize=7.5)
    ax.set_ylabel(f'{uf}/s', fontsize=7.5, color=TXT2)
    ax.margins(y=0.15)

    # C. CIVM: EMG (envoltória) no tempo
    ax = fig.add_subplot(gs[2, 2:])
    _eixo(ax, 'C. Trapézio na CIVM (RMS móvel 50 ms)')
    c = r_civm.emg_ids[0]
    env = np.sqrt(pd.Series(r_civm.emg_seg[c] ** 2).rolling(int(0.05 * r_civm.parametros['fs']), center=True,
                                                             min_periods=1).mean().to_numpy())
    ax.plot(r_civm.t_seg[::k], env[::k], color=AZUL, lw=1.2)
    ax.axvspan(r_civm.t_seg[r_civm.janela[0]], r_civm.t_seg[r_civm.janela[1] - 1], color=ALVO, alpha=0.15, lw=0)
    ax.set_ylabel(r_civm.unidade_emg, fontsize=7.5, color=TXT2)
    ax.set_xlabel('Tempo (s)', fontsize=7.5, color=TXT2)

    if r_fad is not None:
        fr = next(iter(r_fad.forca.values()))
        fs = r_fad.parametros['fs']
        pico = r_civm.mec[f_id]['pico']
        # D. fadiga: força em % CIVM com alvo
        ax = fig.add_subplot(gs[3, :])
        _eixo(ax, f'D. Fadiga — força em % da CIVM (faixa verde = alvo {pt.alvo_pct:g} ± {pt.tolerancia_pct:g}%)')
        kk = max(1, len(fr['fm']) // 4000)
        tt = np.arange(len(fr['fm'])) / fs
        ax.axhspan(pt.alvo_pct - pt.tolerancia_pct, pt.alvo_pct + pt.tolerancia_pct, color=ALVO, alpha=0.15, lw=0)
        ax.plot(tt[::kk], 100 * fr['fm'][::kk] / pico, color=AZUL, lw=1.4)
        ax.axvline(fr['i_ini'] / fs, color=TXT2, ls='--', lw=1)
        ax.text(fr['i_ini'] / fs, 3, ' início do patamar', fontsize=7, color=TXT2)
        ax.set_ylabel('% da CIVM', fontsize=7.5, color=TXT2)
        ax.set_xlabel('Tempo (s)', fontsize=7.5, color=TXT2)
        ax.set_ylim(0, max(80, float(np.nanmax(100 * fr['fm'] / pico)) * 1.1))
        # E/F. MDF e RMS %CIVM por janela
        e = next(iter(r_fad.emg.values()))
        ok = jan_norm['valida']
        t = jan_norm['t_centro_s'] - e['i_ini'] / fs
        for pos, col, rot, tit in ((gs[4, :2], 'MDF_Hz', 'MDF (Hz)', 'E. Frequência mediana (MDF) na fadiga'),
                                   (gs[4, 2:], 'RMS_pct_CIVM', '% da RMS da CIVM', 'F. RMS do trapézio na fadiga')):
            ax = fig.add_subplot(pos)
            _eixo(ax, tit)
            ax.plot(t[ok], jan_norm.loc[ok, col], 'o', color=AZUL, ms=2.5, alpha=0.55)
            if (~ok).any():
                ax.plot(t[~ok], jan_norm.loc[~ok, col], 'x', color='#9a9890', ms=3.5)
            if ok.sum() >= 3:
                b, a = np.polyfit(t[ok], jan_norm.loc[ok, col], 1)
                xx = np.array([t[ok].min(), t[ok].max()])
                ax.plot(xx, a + b * xx, color=TXT, lw=1.5)
                ax.text(0.98, 0.94, f'{b:+.3f} /s', transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
                        color=TXT, bbox=dict(fc='white', ec='none', alpha=0.85))
            ax.set_ylabel(rot, fontsize=7.5, color=TXT2)
            ax.set_xlabel('Tempo desde o início do patamar (s)', fontsize=7.5, color=TXT2)
    return fig


def destaques(r_civm, ex, r_fad, integ, pt) -> list[str]:
    uf = r_civm.unidade_forca
    f_id = r_civm.forca_ids[0]
    m = r_civm.mec[f_id]
    col_t = ex.tdf.columns[0]
    t100 = [i for i in ex.tdf.index if i.startswith('TDF 0–100')]
    tn = [i for i in ex.tdf.index if i.startswith('TDF normalizada 0–100')]
    out = [f'<b>CIVM</b>: pico de {nc.fmt(m["pico"], 1)} {uf}, atingido {nc.fmt(m["tempo_onset_pico_ms"] / 1000, 2)} s após '
           f'o onset; força média na janela de CIVM {nc.fmt(r_civm.df_mec.iloc[1, 0], 1)} {uf}.']
    if t100:
        out.append(f'<b>TDF 0–100 ms</b>: {nc.fmt(ex.tdf.loc[t100[0], col_t], 1)} {uf}/s'
                   + (f' ({nc.fmt(ex.tdf.loc[tn[0], col_t], 1)}% do pico por segundo)' if tn else '') + '.')
    if r_fad is not None:
        e = next(iter(r_fad.emg.values()))['resumo']
        pct = integ.loc['Força média no patamar (% da CIVM)', 'Valor']
        no_alvo = integ.iloc[6, 0]
        out.append(f'<b>Força na fadiga</b>: {nc.fmt(pct, 1)}% da CIVM em média ({"dentro" if abs(pct - pt.alvo_pct) <= pt.tolerancia_pct else "FORA"} '
                   f'do alvo de {pt.alvo_pct:g} ± {pt.tolerancia_pct:g}%), {nc.fmt(no_alvo, 0)}% do tempo do patamar dentro da faixa.')
        sig_mdf = np.isfinite(e['MDF_p']) and e['MDF_p'] < 0.05
        sig_rms = np.isfinite(e['RMS_p']) and e['RMS_p'] < 0.05
        padrao = ('MDF cai e RMS sobe — padrão clássico de fadiga muscular' if (sig_mdf and e['MDF_slope'] < 0 and sig_rms and e['RMS_slope'] > 0)
                  else 'MDF cai sem aumento significativo do RMS' if (sig_mdf and e['MDF_slope'] < 0)
                  else 'RMS sobe sem queda significativa da MDF' if (sig_rms and e['RMS_slope'] > 0)
                  else 'sem tendência significativa de MDF nem de RMS')
        out.append(f'<b>Fadiga eletromiográfica do trapézio</b>: MDF {nc.fmt(e["MDF_ini"], 1)} → {nc.fmt(e["MDF_fim"], 1)} Hz '
                   f'(inclinação {nc.fmt(e["MDF_slope"], 3)} Hz/s, p = {nc.fmt(e["MDF_p"], 3)}; queda pela regressão '
                   f'{nc.fmt(e["Queda_MDF_reg"], 1)}%); RMS {nc.fmt(integ.iloc[8, 0], 1)} → {nc.fmt(integ.iloc[9, 0], 1)}% da '
                   f'RMS da CIVM. Leitura: {padrao}.')
        if e['pct_excluidas'] and e['pct_excluidas'] > 0:
            out.append(f'<b>Artefatos</b>: {nc.fmt(e["pct_excluidas"], 0)}% das janelas da fadiga foram excluídas.')
    n_av = len(r_civm.avisos) + (len(r_fad.avisos) if r_fad is not None else 0)
    out.append(f'<b>Qualidade</b>: {n_av} aviso(s) automático(s)' + (' — veja a seção de controle de qualidade.' if n_av else '.'))
    return out


# ------------------------------------------------------------------ glossário da fadiga e da integração
def glossario_fadiga(pf: fn.Parametros, pt: ParametrosTeste) -> list[tuple[str, str, str, str]]:
    n = lambda x: f'{x:g}'.replace('.', ',')
    G = 'Fadiga (50% da CIVM)'
    I = 'Integração CIVM → fadiga'
    return [
        (G, 'Tara da fadiga', 'Zero da célula de carga no teste de fadiga.',
         f'Média de {n(pf.janela_repouso_s)} s de repouso antes do onset; se o registro já começar com contração, usa-se '
         'a tara da mesma célula medida na CIVM (indicado no relatório).'),
        (G, 'Início do patamar', 'Momento a partir do qual a força está estável no alvo (exclui a subida e o ajuste inicial).',
         f'Primeiro instante após o onset em que a média móvel de {n(pf.media_movel_s)} s entra e permanece por ≥ '
         f'{n(pf.dur_min_estavel_s)} s na faixa média ± {n(pf.k_dp)} DP do patamar; confirmado se CV ≤ '
         f'{n(pf.cv_max_pct)}% e deriva ≤ {n(pf.deriva_max_pct)}%; senão, onset + {n(pf.inicio_reserva_s)} s (sinalizado).'),
        (G, 'Força média, inicial e final; variação; inclinação', 'Quanto de força foi sustentado e se ela caiu ao longo do teste.',
         f'Média no patamar; inicial/final = média dos primeiros/últimos {n(pf.extremos_s)} s; variação = (final − inicial) '
         '÷ inicial × 100; inclinação = regressão linear da força no tempo.'),
        (G, 'CV e CV sem tendência', 'Estabilidade da força: oscilação em torno do alvo.',
         'DP ÷ média × 100; "sem tendência" = DP dos resíduos após remover a reta de regressão ÷ média × 100.'),
        (G, 'Janelas espectrais', 'Trechos curtos em que o EMG é analisado ao longo do patamar.',
         f'{n(pf.janela_espectral_s)} s com {pf.sobreposicao:.0%} de sobreposição (ex.: 0–0,5; 0,25–0,75; 0,5–1,0 s).'),
        (G, 'MDF — frequência mediana (Hz)', 'Frequência que divide a potência do EMG ao meio. Cai com a fadiga '
         '(redução da velocidade de condução e acúmulo de metabólitos).',
         f'Espectro de Welch (segmentos de {n(pf.welch_segmento_s)} s, Hann) em cada janela, na banda '
         f'{n(pf.banda_emg[0])}–{n(pf.banda_emg[1])} Hz; MDF = frequência em que a potência acumulada atinge 50%.'),
        (G, 'MNF — frequência média (Hz)', 'Centro de gravidade do espectro; interpretação semelhante à MDF.',
         'Σ(f × potência) ÷ Σ potência, na mesma banda.'),
        (G, 'Inclinação, R² e p (MDF, MNF, RMS, ENM)', 'Taxa de variação ao longo do patamar; negativa na MDF = fadiga.',
         'Regressão linear vs. tempo nas janelas válidas; p com erro-padrão corrigido pela sobreposição das janelas '
         '(n efetivo = n × (1 − sobreposição)).'),
        (G, 'Queda da MDF pela regressão e pelos extremos (%)', 'Quanto a MDF caiu do início ao fim do patamar.',
         f'Regressão: inclinação ÷ intercepto × duração × 100 (usa todas as janelas — preferível). Extremos: '
         f'(média dos últimos − média dos primeiros {n(pf.extremos_s)} s) ÷ primeiros × 100.'),
        (G, 'RMS na fadiga e variação do RMS', 'Amplitude do EMG; tende a subir com a fadiga (recrutamento de mais unidades '
         'motoras para manter a força).', 'RMS do EMG filtrado em cada janela; variação = (final − inicial) ÷ inicial × 100.'),
        (G, 'ENM na fadiga', 'Força por unidade de ativação; cai quando é preciso mais ativação para a mesma força.',
         'Força média da janela ÷ RMS da janela.'),
        (G, 'Janelas com artefato', 'Trechos contaminados (movimento de cabo/eletrodo, saturação) que distorcem MDF e RMS.',
         f'Excluídas se atingem o limite do conversor (±{n(pf.fundo_escala)}), têm pico > {n(pf.pico_max_dp)} DP robustos '
         f'ou RMS > {n(pf.rms_max_razao)}× a mediana das janelas.'),
        (I, 'Força em % da CIVM', 'Se o voluntário sustentou o alvo de intensidade.',
         f'Força ÷ pico da CIVM × 100; alvo = {n(pt.alvo_pct)}%; % do tempo no alvo = fração do patamar dentro de '
         f'{n(pt.alvo_pct)} ± {n(pt.tolerancia_pct)}% da CIVM.'),
        (I, 'RMS em % da RMS da CIVM', 'Ativação do trapézio relativa ao seu máximo — comparável entre pessoas e dias.',
         'RMS de cada janela da fadiga ÷ RMS do trapézio na janela de CIVM × 100; inicial/final e inclinação como acima.'),
        (I, 'ENM em % da ENM da CIVM', 'Eficiência na fadiga relativa à da contração máxima.',
         'ENM inicial/final da fadiga ÷ ENM da CIVM × 100.'),
    ]


def html_glossario(pc: nc.Parametros, pe: av.ParametrosExtra, pf: fn.Parametros | None, pt: ParametrosTeste) -> str:
    L = [x for x in gl.linhas(pc, pe) if x[0] not in ('Cocontração', 'Assimetria')]
    L = [(('CIVM — ' + g) if g not in ('Pré-processamento', 'Qualidade') else g, v, s, c) for g, v, s, c in L]
    if pf is not None:
        L += glossario_fadiga(pf, pt)
    corpo, atual = [], None
    for grupo, var, sig, calc in L:
        if grupo != atual:
            corpo.append(f'<tr><td colspan="3" style="background:#eaf0f7;font-weight:bold;text-align:left">'
                         f'{html.escape(grupo)}</td></tr>')
            atual = grupo
        corpo.append(f'<tr><td style="text-align:left;width:24%"><b>{html.escape(var)}</b></td>'
                     f'<td style="text-align:left;width:34%">{html.escape(sig)}</td>'
                     f'<td style="text-align:left">{html.escape(calc)}</td></tr>')
    return ('<table><tr><th>Variável</th><th>O que significa</th><th>Como foi calculada</th></tr>'
            + ''.join(corpo) + '</table>')


def linhas_glossario(pc, pe, pf, pt) -> pd.DataFrame:
    L = [x for x in gl.linhas(pc, pe) if x[0] not in ('Cocontração', 'Assimetria')]
    if pf is not None:
        L += glossario_fadiga(pf, pt)
    return pd.DataFrame(L, columns=['Grupo', 'Variável', 'O que significa', 'Como foi calculada'])


# ------------------------------------------------------------------ relatório combinado
def monta_html(r_civm, r_fad, pt: ParametrosTeste, pe: av.ParametrosExtra, meta: dict, nomes: dict) -> str:
    pc = nc.Parametros(**r_civm.parametros)
    pf = fn.Parametros(**r_fad.parametros) if r_fad is not None else None
    ex = av.calcula(r_civm, pe)
    integ, jan_norm, av_int = integra(r_civm, r_fad, pt) if r_fad is not None else (None, None, [])
    agora = datetime.now().strftime('%d/%m/%Y %H:%M')
    ue, uf = r_civm.unidade_emg, r_civm.unidade_forca
    ident = [('Voluntário', meta.get('participante') or '—'), ('Músculo', meta.get('musculo') or 'Trapézio'),
             ('Arquivo da CIVM', nomes.get('civm', '—')), ('Arquivo da fadiga', nomes.get('fadiga') or 'não enviado'),
             ('Avaliador', meta.get('avaliador') or '—'), ('Observações', meta.get('observacoes') or '—')]
    emg_civm = r_civm.df_emg.drop(columns=[c for c in r_civm.df_emg.columns if 'referência' in c])
    sec_fad = ''
    if r_fad is not None:
        mec_f = fi.tabela_mecanica([r_fad], next(iter(r_fad.forca)))
        sec_fad = f"""
<h2 style="page-break-before:always">3. Teste de fadiga a {pt.alvo_pct:g}% da CIVM</h2>
<h3>3.1 Força em relação à CIVM</h3>
{_tabela(integ, 'Variável')}
<h3>3.2 Fadiga mecânica</h3>
{_tabela(mec_f, 'Variável')}
{_figura(f_forca_fad(r_fad), 'Figura 4. Fadiga: força tarada, faixa média ± k·DP do patamar, onset e início do patamar.')}
<h3>3.3 Fadiga eletromiográfica do trapézio</h3>
{_tabela(r_fad.df_emg, 'Músculo')}
{_tabela(r_fad.df_emg_extra, 'Canal')}
{_figura(fig_tendencia(r_fad, 'MDF_Hz', 'MDF (Hz)'), 'Figura 5. MDF por janela no patamar e reta de regressão (x = janela excluída).')}
{_figura(fig_tendencia(r_fad, 'RMS', f'RMS ({ue})'), 'Figura 6. RMS por janela no patamar e reta de regressão.')}
{_figura(f_emg_fad(r_fad), 'Figura 7. EMG filtrado na fadiga; verde = patamar; vermelho = janelas com artefato.')}
"""
    avisos = list(r_civm.avisos) + ([f'[Fadiga] {a}' for a in r_fad.avisos] if r_fad is not None else []) + av_int
    avisos_html = ''.join(f'<div class="aviso">{html.escape(a)}</div>' for a in avisos) or \
        '<div class="ok">Nenhum aviso de qualidade foi gerado nas verificações automáticas.</div>'
    atenu = [r_civm.atenuacao_notch.rename('CIVM')] + ([r_fad.atenuacao_notch.rename('Fadiga')] if r_fad is not None else [])
    atenu_df = pd.concat(atenu, axis=1) if any(len(a) for a in atenu) else pd.DataFrame()
    metodos = [(f'CIVM — {g} — {n}', v) for g, n, v in nc.descricao_parametros(pc)]
    if pf is not None:
        metodos += [(f'Fadiga — {g} — {n}', v) for g, n, v in fn.descricao_parametros(pf)]
    resumo = ''.join(f'<li>{d}</li>' for d in destaques(r_civm, ex, r_fad, integ, pt))
    f_tdf = av.fig_tdf(ex.tdf, uf)
    return f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Trapézio — CIVM e fadiga</title><style>{CSS.replace('Relatório de análise EMG + força — CIVM', 'Trapézio — CIVM e fadiga')}</style></head><body>
<h1>Relatório — trapézio: CIVM e fadiga a {pt.alvo_pct:g}% da CIVM</h1>
<div class="sub">1 eletrodo de EMG + 1 célula de carga · gerado em {agora} · versão {nc.APP_VERSION}</div>

<h2>1. Identificação</h2>
{_tabela_txt(ident)}

<h2>2. Contração isométrica voluntária máxima (CIVM)</h2>
<h3>2.1 Mecânica</h3>
{_tabela(r_civm.df_mec, 'Variável')}
{_figura(f_forca_civm(r_civm), 'Figura 1. CIVM: força tarada, onset, pico e janela de CIVM.')}
<h3>2.2 Taxa de desenvolvimento de força (TDF)</h3>
{_tabela(ex.tdf, 'Variável')}
{_figura(fig_onset(r_civm), 'Figura 2. Ampliação do onset com os instantes de 50, 100 e 200 ms.')}
{_figura(f_tdf, 'Figura 2a. TDF nas janelas iniciais.') if f_tdf else ''}
<h3>2.3 Eletromiografia do trapézio na CIVM</h3>
{_tabela(emg_civm, 'Músculo')}
{_figura(f_emg_civm(r_civm), 'Figura 3. EMG filtrado na CIVM; faixa verde = janela de CIVM.')}
{sec_fad}
<h2>4. Controle de qualidade</h2>
{avisos_html}
<h3>Auditoria espectral (atenuação em {pc.f_notch:g} ± 2 Hz, dB)</h3>
{_tabela(atenu_df, 'Canal') if len(atenu_df) else ''}

<h2>5. Métodos e rastreabilidade</h2>
{_tabela_txt(metodos)}
{_tabela_txt(list(nc.versoes().items()))}

<h2 style="page-break-before:always">6. Glossário: o que cada variável significa e como foi calculada</h2>
{html_glossario(pc, pe, pf, pt)}

<h2 style="page-break-before:always">7. Resumo visual dos resultados</h2>
<ul style="font-size:8.4pt;margin:0 0 6pt 14pt;padding:0">{resumo}</ul>
{_figura(fig_resumo(r_civm, ex, r_fad, integ, jan_norm, pt), 'Painel-resumo. Verde = janela de CIVM / faixa do alvo; x = janela excluída por artefato.')}
</body></html>"""
