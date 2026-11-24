"""Figuras (matplotlib) usadas na interface e no relatório em PDF — número de canais variável."""
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from nucleo import lado_de, nome_canal

CORES_LADO = {'Dir': '#1f77b4', 'Esq': '#d62728', '': '#2f6f4f'}


def _cor(canal_id: str) -> str:
    return CORES_LADO.get(lado_de(canal_id), '#2f6f4f')


def _passo(n, alvo=20000):
    return max(1, n // alvo)


def _grade(n_paineis: int, ncols: int = 2):
    """Cria uma grade com n painéis (2 colunas) e esconde os eixos que sobrarem."""
    ncols = 1 if n_paineis == 1 else ncols
    nrows = math.ceil(n_paineis / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, max(2.4, 2.3 * nrows)), sharex=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n_paineis:]:
        ax.set_visible(False)
    return fig, axes[:n_paineis], ncols


def _ordem_emg(res):
    """Ordena os canais de EMG por músculo e lado (Dir à esquerda, Esq à direita) quando possível."""
    return list(res.emg_ids)


def fig_forca(res):
    """Força tarada de cada canal, onset, pico, limiar e janela de CIVM."""
    uf = res.unidade_forca
    fig, ax = plt.subplots(figsize=(10, 4))
    k = _passo(len(res.t_seg))
    handles = []
    for f_id in res.forca_ids:
        f, m, cor = res.forcas[f_id], res.mec[f_id], _cor(f_id)
        ax.plot(res.t_seg[::k], f[::k], color=cor, lw=1)
        ax.axvline(res.t_seg[m['idx_onset']], color=cor, ls=':', lw=1.2)
        ax.plot(res.t_seg[m['idx_pico']], f[m['idx_pico']], 'o', color=cor, mec='k', zorder=5)
        ax.axhline(m['limiar'], color=cor, ls='--', lw=0.6, alpha=0.6)
        handles.append(Line2D([], [], color=cor, label=nome_canal(f_id)))
    ini, fim = res.janela
    ax.axvspan(res.t_seg[ini], res.t_seg[fim - 1], color='green', alpha=0.13)
    handles += [Line2D([], [], color='k', ls=':', label='Onset'),
                Line2D([], [], color='k', ls='--', lw=0.7, label='Limiar de onset'),
                Line2D([], [], color='k', marker='o', ls='', mfc='w', label='Pico'),
                Patch(color='green', alpha=0.25, label='Janela de CIVM')]
    fig.legend(handles=handles, ncol=len(handles), fontsize=8, loc='lower center', frameon=False)
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel(f'Força tarada ({uf})')
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def fig_onset(res):
    """Ampliação em torno do onset de cada canal de força, com as janelas de RFD marcadas."""
    p = res.parametros
    fs = p['fs']
    uf = res.unidade_forca
    n = len(res.forca_ids)
    fig, axes = plt.subplots(1, n, figsize=(10 if n > 1 else 6, 3.4), squeeze=False)
    ws = list(p['janelas_rfd_ms'])
    for ax, f_id in zip(axes[0], res.forca_ids):
        f, m, cor = res.forcas[f_id], res.mec[f_id], _cor(f_id)
        a = max(0, m['idx_onset'] - int(0.10 * fs))
        b = min(len(f), m['idx_onset'] + int((max(ws) + 100) / 1000 * fs))
        t = (np.arange(a, b) - m['idx_onset']) / fs * 1000
        ax.plot(t, f[a:b], color=cor, lw=1.2)
        ax.axhline(m['limiar'], color='gray', ls='--', lw=0.7)
        ax.axvline(0, color='k', ls=':', lw=1)
        for w in ws:
            idx = m['idx_onset'] + int(round(w / 1000 * fs))
            if idx < len(f):
                ax.axvline(w, color='gray', lw=0.6, alpha=0.6)
                ax.plot(w, f[idx], 's', color=cor, mec='k', ms=4)
                ax.text(w, 0.97, f'{w} ms', transform=ax.get_xaxis_transform(), fontsize=7,
                        ha='center', va='top', color='gray', bbox=dict(fc='w', ec='none', pad=0.5))
        ax.set_title(f'{nome_canal(f_id)} — ampliação do onset', fontsize=9)
        ax.set_xlabel('Tempo a partir do onset (ms)')
        ax.set_ylabel(f'Força ({uf})')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_emg(res):
    """EMG filtrado (trecho analisado) de cada canal, com a janela de CIVM sombreada."""
    ue = res.unidade_emg
    canais = _ordem_emg(res)
    fig, axes, ncols = _grade(len(canais))
    ini, fim = res.janela
    k = _passo(len(res.t_seg), 15000)
    for i, (ax, canal) in enumerate(zip(axes, canais)):
        ax.plot(res.t_seg[::k], res.emg_seg[canal][::k], color=_cor(canal), lw=0.5)
        ax.axvspan(res.t_seg[ini], res.t_seg[fim - 1], color='green', alpha=0.13)
        ax.set_title(nome_canal(canal), fontsize=9)
        ax.grid(alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel(f'EMG filtrado ({ue})')
        ax.tick_params(labelbottom=True)
        ax.set_xlabel('Tempo (s)')
    fig.tight_layout()
    return fig


def fig_psd(res):
    """PSD (Welch) bruta x filtrada de cada canal de EMG."""
    f0 = res.parametros['f_notch']
    canais = _ordem_emg(res)
    fig, axes, ncols = _grade(len(canais))
    for i, (ax, canal) in enumerate(zip(axes, canais)):
        f, pb, pf = res.psd[canal]
        ax.plot(f, 10 * np.log10(pb + 1e-30), color='#ff7f0e', alpha=0.7, lw=0.9, label='Bruto (sem offset)')
        ax.plot(f, 10 * np.log10(pf + 1e-30), color='#1f77b4', lw=0.9, label='Filtrado')
        ax.axvline(f0, color='red', ls='--', lw=0.7)
        ax.set_xlim(0, min(600, f[-1]))
        ax.set_title(nome_canal(canal), fontsize=9)
        ax.grid(alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel('PSD (dB/Hz)')
        ax.tick_params(labelbottom=True)
        ax.set_xlabel('Frequência (Hz)')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig
