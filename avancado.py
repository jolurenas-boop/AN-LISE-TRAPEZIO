"""Análises adicionais da CIVM: COCONTRAÇÃO agonista/antagonista, TDF (taxa de desenvolvimento de força)
e quadro de ASSIMETRIAS bilaterais.

Calculadas sobre o Resultado do núcleo (força tarada, onset, EMG filtrado, janela de CIVM), sem alterar o
processamento principal.

Cocontração (por lado; agonista = bíceps na flexão do cotovelo, antagonista = tríceps):
  * Razão de coativação (%) = RMS do antagonista ÷ RMS do agonista × 100, na janela.
  * Índice de cocontração de Falconer & Winter (1985) (%) = 2 × Σ min(envA, envB) ÷ Σ (envA + envB) × 100,
    com envoltórias RMS móveis (padrão 50 ms).
  Janelas: janela de CIVM e fases iniciais 0–50, 0–100 e 0–200 ms após o onset da força do mesmo lado.
  Sem RMS de referência, os índices usam µV (dependem de ganho/eletrodo de cada músculo); com a RMS de CIVM de
  cada músculo informada (ex.: tríceps numa CIVM de extensão), as envoltórias são normalizadas (% da CIVM),
  que é a forma recomendada.

TDF (por lado), para cada janela w após o onset:
  * TDF absoluta (kgf/s) = [F(onset + w) − F(onset)] ÷ w   (= RFD do núcleo)
  * TDF normalizada (%Fpico/s) = TDF ÷ pico × 100
  * Força em w (kgf) e impulso 0–w (kgf·s) = ∫ [F − F(onset)] dt
  * TDF máxima (kgf/s) = maior inclinação em janelas deslizantes de 20 ms nos primeiros 200 ms.

Assimetria (D vs. E), para cada variável:
  * Assimetria (%) = 100 × (D − E) ÷ máx(|D|, |E|)   (mesma do núcleo)
  * Índice de simetria (%) = 100 × (D − E) ÷ [(D + E)/2]
  * LSI (%) = 100 × menor ÷ maior
  * Sinalização quando |Assimetria| > limiar (padrão 10%).
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nucleo as nc

ROT_LADO = {'Dir': 'Lado D', 'Esq': 'Lado E'}
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')   # NumPy 2.x / 1.x


@dataclass(frozen=True)
class ParametrosExtra:
    agonista: str = 'Bíceps braquial'
    antagonista: str = 'Tríceps braquial'
    tarefa: str = 'Flexão do cotovelo'
    envoltoria_ms: float = 50.0
    rms_ref: tuple = ()               # ((nome, lado, RMS de referência), ...) — opcional, para normalizar
    limiar_assimetria_pct: float = 10.0
    janela_tdf_max_ms: float = 20.0


# ------------------------------------------------------------------ utilidades
def assimetria(d, e) -> float:
    if not (np.isfinite(d) and np.isfinite(e)) or max(abs(d), abs(e)) == 0:
        return np.nan
    return 100 * (d - e) / max(abs(d), abs(e))


def indice_simetria(d, e) -> float:
    if not (np.isfinite(d) and np.isfinite(e)) or (d + e) == 0:
        return np.nan
    return 100 * (d - e) / ((d + e) / 2)


def lsi(d, e) -> float:
    if not (np.isfinite(d) and np.isfinite(e)) or max(abs(d), abs(e)) == 0:
        return np.nan
    return 100 * min(abs(d), abs(e)) / max(abs(d), abs(e))


def _rms_movel(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    return np.sqrt(pd.Series(np.asarray(x, float) ** 2).rolling(n, center=True, min_periods=1).mean().to_numpy())


def _forca_do_lado(res, lado):
    for f_id in res.forca_ids:
        if nc.lado_de(f_id) == lado:
            return f_id
    return res.forca_ids[0] if len(res.forca_ids) == 1 else None


def _ref(pe: ParametrosExtra, nome, lado):
    for n, l, v in pe.rms_ref:
        if n == nome and l == lado and v and np.isfinite(v) and v > 0:
            return float(v)
    return None


# ------------------------------------------------------------------ TDF
def tdf(res) -> pd.DataFrame:
    p = res.parametros
    fs, uf = p['fs'], res.unidade_forca
    cols, dados = [], {}
    for f_id in res.forca_ids:
        lado = nc.lado_de(f_id)
        col = ROT_LADO.get(lado, nc.nome_canal(f_id))
        cols.append(col)
        f, m = res.forcas[f_id], res.mec[f_id]
        on, pico = m['idx_onset'], m['pico']
        d = {}
        for w in p['janelas_rfd_ms']:
            i = on + int(round(w / 1000 * fs))
            ok = i < len(f) and on > 0
            delta = float(f[i] - f[on]) if ok else np.nan
            d[f'TDF 0–{w} ms ({uf}/s)'] = delta / (w / 1000) if ok else np.nan
            d[f'TDF normalizada 0–{w} ms (%Fpico/s)'] = 100 * (delta / (w / 1000)) / pico if ok and pico else np.nan
            d[f'Força em {w} ms ({uf})'] = delta
            d[f'Impulso 0–{w} ms ({uf}·s)'] = float(_trapz(f[on:i + 1] - f[on], dx=1 / fs)) if ok else np.nan
        n_w = max(2, int(round(20 / 1000 * fs)))
        fim = min(len(f) - n_w, on + int(0.2 * fs))
        if on > 0 and fim > on:
            incl = (f[on + n_w:fim + n_w] - f[on:fim]) / (n_w / fs)
            d[f'TDF máxima 0–200 ms ({uf}/s)'] = float(incl.max())
            d['Instante da TDF máxima (ms após o onset)'] = float(np.argmax(incl) / fs * 1000)
        else:
            d[f'TDF máxima 0–200 ms ({uf}/s)'] = np.nan
            d['Instante da TDF máxima (ms após o onset)'] = np.nan
        d[f'Pico de força ({uf})'] = pico
        dados[col] = d
    df = pd.DataFrame(dados)[cols]
    if len(cols) == 2:
        df['Assimetria (%)'] = [assimetria(a, b) if 'Instante' not in i else np.nan
                                for i, (a, b) in zip(df.index, df[cols].to_numpy())]
    return df


# ------------------------------------------------------------------ cocontração
def cocontracao(res, pe: ParametrosExtra) -> tuple[pd.DataFrame, dict, list[str]]:
    """Retorna (tabela, envoltórias para figura, avisos)."""
    p = res.parametros
    fs = p['fs']
    n_env = int(round(pe.envoltoria_ms / 1000 * fs))
    avisos, env_fig, dados, cols = [], {}, {}, []
    for lado in ('Dir', 'Esq'):
        ag, an = nc.id_canal(pe.agonista, lado), nc.id_canal(pe.antagonista, lado)
        if ag not in res.emg_seg or an not in res.emg_seg:
            continue
        f_id = _forca_do_lado(res, lado)
        if f_id is None:
            avisos.append(f'Cocontração lado {nc.NOME_LADO[lado]}: sem célula de carga para o onset.')
            continue
        r_ag, r_an = _ref(pe, pe.agonista, lado), _ref(pe, pe.antagonista, lado)
        norm = r_ag is not None and r_an is not None
        xa, xb = res.emg_seg[ag], res.emg_seg[an]
        if norm:
            xa, xb = xa / r_ag * 100, xb / r_an * 100
        ea, eb = _rms_movel(xa, n_env), _rms_movel(xb, n_env)
        on = res.mec[f_id]['idx_onset']
        janelas = {'janela de CIVM': res.janela}
        for w in p['janelas_rfd_ms']:
            janelas[f'0–{w} ms'] = (on, on + int(round(w / 1000 * fs)))
        col = ROT_LADO[lado]
        cols.append(col)
        d = {}
        for rot, (a, b) in janelas.items():
            if on <= 0 and rot != 'janela de CIVM' or b > len(xa):
                razao = cci = np.nan
            else:
                rms_a = float(np.sqrt(np.mean(xa[a:b] ** 2)))
                rms_b = float(np.sqrt(np.mean(xb[a:b] ** 2)))
                razao = 100 * rms_b / rms_a if rms_a > 0 else np.nan
                soma = float(np.sum(ea[a:b] + eb[a:b]))
                cci = 200 * float(np.sum(np.minimum(ea[a:b], eb[a:b]))) / soma if soma > 0 else np.nan
            d[f'Razão antagonista/agonista — {rot} (%)'] = razao
            d[f'Índice de cocontração (Falconer & Winter) — {rot} (%)'] = cci
        dados[col] = d
        env_fig[lado] = {'ea': ea, 'eb': eb, 'on': on, 'norm': norm, 'f_id': f_id}
        if not norm:
            avisos.append(f'Cocontração lado {nc.NOME_LADO[lado]}: índices em µV, sem normalização pela CIVM de cada '
                          f'músculo — dependem do ganho/posição dos eletrodos. Informe a RMS de referência para normalizar.')
    if not dados:
        return pd.DataFrame(), env_fig, avisos
    ordem_lin = list(next(iter(dados.values())).keys())
    df = pd.DataFrame(dados).loc[ordem_lin, cols]
    if len(cols) == 2:
        df['Assimetria (%)'] = [assimetria(a, b) for a, b in df[cols].to_numpy()]
    return df, env_fig, avisos


# ------------------------------------------------------------------ quadro de assimetrias
def quadro_assimetrias(res, df_tdf: pd.DataFrame, df_coc: pd.DataFrame, pe: ParametrosExtra) -> pd.DataFrame:
    uf, ue = res.unidade_forca, res.unidade_emg
    linhas = []

    def add(var, d, e):
        a = assimetria(d, e)
        linhas.append({'Variável': var, 'Lado D': d, 'Lado E': e, 'Assimetria (%)': a,
                       'Índice de simetria (%)': indice_simetria(d, e), 'LSI (%)': lsi(d, e),
                       'Lado maior': ('D' if d > e else 'E') if np.isfinite(d) and np.isfinite(e) and d != e else '—',
                       f'|Assimetria| > {pe.limiar_assimetria_pct:g}%': ('SIM' if abs(a) > pe.limiar_assimetria_pct
                                                                          else 'não') if np.isfinite(a) else '—'})

    mec = res.df_mec
    if 'Direito' in mec.columns and 'Esquerdo' in mec.columns:
        for var in mec.index:
            if any(k in var for k in ('Instante', 'Tempo onset')):
                continue
            add(var.replace('RFD', 'TDF'), float(mec.loc[var, 'Direito']), float(mec.loc[var, 'Esquerdo']))
    if {'Lado D', 'Lado E'} <= set(df_tdf.columns):
        for var in df_tdf.index:
            if ('normalizada' in var or 'Impulso' in var or 'máxima' in var) and 'Instante' not in var:
                add(var, float(df_tdf.loc[var, 'Lado D']), float(df_tdf.loc[var, 'Lado E']))
    for nome in dict.fromkeys(nc.lado_de(c) and c.split('|')[0] for c in res.emg_ids):
        if not nome:
            continue
        cd, ce = nc.id_canal(nome, 'Dir'), nc.id_canal(nome, 'Esq')
        nd, ne = nc.nome_canal(cd), nc.nome_canal(ce)
        if nd in res.df_emg.index and ne in res.df_emg.index:
            cr = [c for c in res.df_emg.columns if c.startswith('RMS')][0]
            ce_ = [c for c in res.df_emg.columns if c.startswith('Eficiência')][0]
            add(f'RMS {nome} ({ue})', float(res.df_emg.loc[nd, cr]), float(res.df_emg.loc[ne, cr]))
            add(f'ENM {nome} ({uf}/{ue})', float(res.df_emg.loc[nd, ce_]), float(res.df_emg.loc[ne, ce_]))
    if {'Lado D', 'Lado E'} <= set(df_coc.columns):
        for var in df_coc.index:
            add(var, float(df_coc.loc[var, 'Lado D']), float(df_coc.loc[var, 'Lado E']))
    return pd.DataFrame(linhas).set_index('Variável') if linhas else pd.DataFrame()


# ------------------------------------------------------------------ figuras
def fig_cocontracao(res, env_fig: dict, pe: ParametrosExtra):
    lados = list(env_fig)
    if not lados:
        return None
    fs = res.parametros['fs']
    fig, axes = plt.subplots(len(lados), 1, figsize=(10, 2.9 * len(lados)), squeeze=False)
    for ax, lado in zip(axes[:, 0], lados):
        e = env_fig[lado]
        ini = max(0, e['on'] - int(0.3 * fs))
        fim = min(len(e['ea']), res.janela[1] + int(0.3 * fs))
        t = res.t_seg[ini:fim]
        ea, eb = e['ea'][ini:fim], e['eb'][ini:fim]
        ax.plot(t, ea, color='#1f4e79', lw=1.1, label=f'{pe.agonista} (agonista)')
        ax.plot(t, eb, color='#c0581c', lw=1.1, label=f'{pe.antagonista} (antagonista)')
        ax.fill_between(t, 0, np.minimum(ea, eb), color='#c0581c', alpha=0.25, lw=0, label='Área comum (cocontração)')
        ax.axvline(res.t_seg[e['on']], color='k', ls=':', lw=1, label='Onset da força')
        ax.axvspan(res.t_seg[res.janela[0]], res.t_seg[res.janela[1] - 1], color='green', alpha=0.08,
                   label='Janela de CIVM')
        ax.set_title(f'Lado {nc.NOME_LADO[lado]} — envoltória RMS ({pe.envoltoria_ms:g} ms)', fontsize=9)
        ax.set_ylabel('% da CIVM de referência' if e['norm'] else f'RMS ({res.unidade_emg})')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3, loc='upper left')
    axes[-1, 0].set_xlabel('Tempo (s)')
    fig.tight_layout()
    return fig


def fig_assimetrias(quadro: pd.DataFrame, limiar: float):
    q = quadro['Assimetria (%)'].dropna()
    if q.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(q) + 1.2))
    cores = ['#c0392b' if abs(v) > limiar else '#7f8c8d' for v in q.values]
    ax.barh(range(len(q)), q.values, color=cores)
    ax.axvline(0, color='k', lw=0.8)
    for x in (-limiar, limiar):
        ax.axvline(x, color='#c0392b', ls='--', lw=0.8)
    ax.set_yticks(range(len(q)))
    ax.set_yticklabels([s if len(s) < 70 else s[:67] + '…' for s in q.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Assimetria D–E (%)  ·  positivo = direito maior  ·  tracejado = limiar', fontsize=8)
    ax.grid(alpha=0.3, axis='x')
    fig.tight_layout()
    return fig


def fig_tdf(df_tdf: pd.DataFrame, uf: str):
    lin = [i for i in df_tdf.index if i.startswith('TDF 0–')]
    cols = [c for c in ('Lado D', 'Lado E') if c in df_tdf.columns] or [df_tdf.columns[0]]
    if not lin:
        return None
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(lin))
    larg = 0.8 / len(cols)
    for j, c in enumerate(cols):
        ax.bar(x + (j - (len(cols) - 1) / 2) * larg, df_tdf.loc[lin, c].astype(float), width=larg,
               color={'Lado D': '#1f77b4', 'Lado E': '#d62728'}.get(c, '#555'), label=c)
    ax.set_xticks(x)
    ax.set_xticklabels([i.split(' (')[0].replace('TDF ', '') for i in lin])
    ax.set_ylabel(f'TDF ({uf}/s)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ pacote completo
@dataclass
class Extras:
    tdf: pd.DataFrame
    coc: pd.DataFrame
    assim: pd.DataFrame
    env: dict
    avisos: list = field(default_factory=list)


def calcula(res, pe: ParametrosExtra) -> Extras:
    d_tdf = tdf(res)
    d_coc, env, av = cocontracao(res, pe)
    return Extras(d_tdf, d_coc, quadro_assimetrias(res, d_tdf, d_coc, pe), env, av)


def secoes_html(res, ex: Extras, pe: ParametrosExtra, _tabela, _figura) -> dict:
    """Blocos HTML para o relatório: {'mec': ..., 'emg': ...}."""
    uf = res.unidade_forca
    f_tdf = fig_tdf(ex.tdf, uf)
    mec = (f'<h3>2.1 Taxa de desenvolvimento de força (TDF)</h3>{_tabela(ex.tdf, cabecalho_indice="Variável")}'
           '<p class="nota">TDF = ΔForça/Δt do onset até 50, 100 e 200 ms; normalizada = TDF ÷ pico × 100; impulso = área '
           'da força acima do valor no onset; TDF máxima = maior inclinação em janelas de 20 ms nos primeiros 200 ms.</p>'
           + (_figura(f_tdf, 'Figura 2a. TDF nas janelas iniciais, por lado.') if f_tdf else ''))
    norm = any(e['norm'] for e in ex.env.values())
    f_coc = fig_cocontracao(res, ex.env, pe)
    coc = ''
    if len(ex.coc):
        coc = (f'<h3>3.1 Cocontração — {html.escape(pe.tarefa)} (agonista: {html.escape(pe.agonista)}; '
               f'antagonista: {html.escape(pe.antagonista)})</h3>'
               + _tabela(ex.coc, cabecalho_indice='Índice')
               + '<p class="nota">Razão = RMS do antagonista ÷ RMS do agonista × 100. Índice de Falconer &amp; Winter = '
                 '2 × área comum das envoltórias ÷ soma das áreas × 100 (0% = sem cocontração; 100% = ativação igual). '
               + ('Envoltórias normalizadas pela RMS de CIVM de cada músculo.' if norm else
                  '<b>Sem normalização</b>: valores em µV, sensíveis a ganho e posição dos eletrodos — para comparar '
                  'músculos e lados, informe a RMS de CIVM de cada músculo (ex.: tríceps numa CIVM de extensão).')
               + '</p>' + (_figura(f_coc, 'Figura 3a. Envoltórias do agonista e do antagonista; área sombreada = '
                                          'cocontração.') if f_coc else ''))
    f_as = fig_assimetrias(ex.assim, pe.limiar_assimetria_pct) if len(ex.assim) else None
    assim = ''
    if len(ex.assim):
        assim = (f'<h3>3.2 Resumo das assimetrias bilaterais</h3>{_tabela(ex.assim, cabecalho_indice="Variável")}'
                 '<p class="nota">Assimetria = 100 × (D − E) ÷ máx(|D|, |E|); índice de simetria = 100 × (D − E) ÷ média; '
                 f'LSI = menor ÷ maior × 100. Sinalizado quando |assimetria| &gt; {pe.limiar_assimetria_pct:g}%.</p>'
                 + (_figura(f_as, 'Figura 3b. Assimetria D–E de cada variável (vermelho = acima do limiar).') if f_as else ''))
    return {'mec': mec, 'emg': coc + assim}
