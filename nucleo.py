"""Núcleo de processamento EMG + células de carga em CIVM — versão GENÉRICA.

Os canais (músculos e células de carga) não são fixos: cada um é descrito por
(nome, lado, tipo, coluna), com lado em {'Dir', 'Esq', ''} ('' = único/sem lado) e tipo em {'EMG', 'Força'}.

Regras para os canais de força:
  * 1 canal de força (qualquer lado)  -> análise unilateral / força única;
  * 2 canais de força, um 'Dir' e um 'Esq' -> análise bilateral, com assimetria D–E.
A janela de CIVM é a de maior força média da SOMA dos canais de força.
A eficiência neuromuscular (ENM) de cada músculo usa a força do mesmo lado; se o músculo não tem lado,
ou se só há um canal de força, usa a força total.

Sem dependência do Streamlit: pode ser usado em scripts/notebooks e testado isoladamente.
Todas as funções que registram problemas de qualidade acrescentam mensagens à lista `avisos`
(em vez de imprimir), para que a interface e o relatório em PDF possam exibi-las.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos, welch

APP_VERSION = '1.0.0-biceps-triceps'
KGF_PARA_N = 9.80665
TOL_PICO_JANELA_S = 0.25    # tolerância para o pico estar fora da janela sem gerar aviso

LADOS_VALIDOS = ('Dir', 'Esq', '')
TIPOS_VALIDOS = ('EMG', 'Força')
NOME_LADO = {'Dir': 'Direito', 'Esq': 'Esquerdo', '': ''}
SEPARADORES = {'tab': '\t', 'ponto_virgula': ';', 'espaco': r'\s+', 'virgula': ','}

# Configuração padrão: bíceps e tríceps braquial (D/E) + 2 células de carga (D/E).
# Cada canal: (nome, lado, tipo, coluna do arquivo, a partir de 1). A coluna 1 é o Tempo.
CANAIS_PADRAO = (
    ('Bíceps braquial', 'Dir', 'EMG', 2), ('Bíceps braquial', 'Esq', 'EMG', 3),
    ('Tríceps braquial', 'Dir', 'EMG', 4), ('Tríceps braquial', 'Esq', 'EMG', 5),
    ('Carga', 'Dir', 'Força', 6), ('Carga', 'Esq', 'Força', 7),
)


def id_canal(nome: str, lado: str) -> str:
    """Identificador interno estável (ex.: 'Vasto lateral', 'Dir' -> 'Vasto lateral|Dir')."""
    return f'{nome}|{lado}' if lado else nome


def nome_canal(canal_id: str) -> str:
    """Nome de exibição: 'Vasto lateral|Dir' -> 'Vasto lateral Direito'."""
    nome, _, lado = canal_id.partition('|')
    adj = NOME_LADO[lado]
    if adj and nome.split()[0].lower().endswith('a'):        # concordância: "Carga Direita"
        adj = adj[:-1] + 'a'
    return f'{nome} {adj}'.strip()


def lado_de(canal_id: str) -> str:
    return canal_id.partition('|')[2]


def canais_do_tipo(p: 'Parametros', tipo: str) -> list[str]:
    return [id_canal(n, l) for n, l, t, _ in p.canais if t == tipo]


def rotulo_forca(p: 'Parametros', fid: str) -> str:
    """Rótulo da coluna da tabela de mecânica. Com força bilateral D/E usa 'Direito'/'Esquerdo'
    (mantém compatibilidade com o portal que lê os PDFs)."""
    lado = lado_de(fid)
    return NOME_LADO[lado] if lado else nome_canal(fid)


def config_para_json(p: 'Parametros') -> bytes:
    cfg = {'coluna_tempo': p.coluna_tempo,
           'canais': [{'nome': n, 'lado': l, 'tipo': t, 'coluna': c} for n, l, t, c in p.canais]}
    return json.dumps(cfg, ensure_ascii=False, indent=2).encode('utf-8')


def config_de_json(conteudo: bytes) -> dict:
    """Lê uma configuração salva. Retorna {'coluna_tempo': int, 'canais': tuple(...)}; valida o formato."""
    try:
        cfg = json.loads(conteudo.decode('utf-8-sig'))
        canais = tuple((str(c['nome']).strip(), str(c.get('lado') or '').strip(), str(c['tipo']).strip(),
                        int(c['coluna'])) for c in cfg['canais'])
        return {'coluna_tempo': int(cfg.get('coluna_tempo', 1)), 'canais': canais}
    except Exception as e:  # noqa: BLE001
        raise ValueError(f'Arquivo de configuração inválido: {e}') from e


# ==============================================================================
# PARÂMETROS
# ==============================================================================
@dataclass(frozen=True)
class Parametros:
    # aquisição / formato do arquivo
    fs: float = 2000.0
    separador: str = 'auto'                 # auto | tab | ponto_virgula | espaco | virgula
    decimal: str = '.'
    linhas_cabecalho: int = 0
    coluna_tempo: int = 1                   # coluna (1-based) do tempo
    canais: tuple = CANAIS_PADRAO           # ((nome, lado, tipo, coluna), ...)
    # unidades
    unidade_forca_entrada: str = 'kgf'
    unidade_forca_saida: str = 'kgf'
    unidade_emg: str = 'mV'
    fator_emg: float = 1.0
    # recorte da análise (s, relativo ao início do registro); fim <= 0 => até o final
    recorte_ini_s: float = 0.0
    recorte_fim_s: float = 0.0
    # EMG
    banda_emg: tuple = (20.0, 500.0)
    ordem_emg: int = 4
    usar_notch: bool = True
    f_notch: float = 60.0
    q_notch: float = 30.0
    # força
    fc_forca: float = 20.0
    ordem_forca: int = 4
    # janela de CIVM, linha de base e onset
    janela_civm_s: float = 1.0
    janela_manual_ini_s: float | None = None    # None => automática (maior força média)
    janela_base_s: float = 0.5
    onset_pct_pico: float = 0.02
    n_sd_ruido: float = 3.0
    # RFD
    janelas_rfd_ms: tuple = (50, 100, 200)


def valida_parametros(p: Parametros) -> None:
    nyq = p.fs / 2
    if p.fs <= 0:
        raise ValueError('A frequência de amostragem deve ser positiva.')
    lo, hi = p.banda_emg
    if not (0 < lo < hi < nyq):
        raise ValueError(f'Banda do EMG inválida: é preciso 0 < f_baixa < f_alta < fs/2 ({nyq:g} Hz).')
    if p.usar_notch and not (0 < p.f_notch < nyq):
        raise ValueError(f'Frequência do notch inválida: é preciso 0 < f < fs/2 ({nyq:g} Hz).')
    if not (0 < p.fc_forca < nyq):
        raise ValueError(f'Corte do filtro de força inválido: é preciso 0 < fc < fs/2 ({nyq:g} Hz).')
    if p.janela_civm_s <= 0 or p.janela_base_s <= 0:
        raise ValueError('As janelas de CIVM e de linha de base devem ser positivas.')
    if not (0 < p.onset_pct_pico < 1):
        raise ValueError('O limiar de onset deve estar entre 0 e 100% do pico.')
    if not p.janelas_rfd_ms:
        raise ValueError('Selecione ao menos uma janela de RFD.')
    valida_canais(p)
    if p.unidade_forca_entrada not in ('kgf', 'N') or p.unidade_forca_saida not in ('kgf', 'N'):
        raise ValueError('Unidade de força deve ser kgf ou N.')


def valida_canais(p: Parametros) -> None:
    if not p.canais:
        raise ValueError('Cadastre ao menos um canal de EMG e um de força.')
    ids, colunas = set(), [p.coluna_tempo]
    for nome, lado, tipo, col in p.canais:
        if not nome:
            raise ValueError('Todo canal precisa de um nome.')
        if '|' in nome:
            raise ValueError(f'O nome "{nome}" não pode conter o caractere "|".')
        if lado not in LADOS_VALIDOS:
            raise ValueError(f'Lado inválido em "{nome}": use Dir, Esq ou vazio.')
        if tipo not in TIPOS_VALIDOS:
            raise ValueError(f'Tipo inválido em "{nome}": use EMG ou Força.')
        if int(col) < 1:
            raise ValueError(f'Coluna inválida em "{nome}": as colunas começam em 1.')
        chave = id_canal(nome, lado)
        if chave in ids:
            raise ValueError(f'Canal repetido: "{nome_canal(chave)}" (nome + lado devem ser únicos).')
        ids.add(chave)
        colunas.append(int(col))
    if len(set(colunas)) != len(colunas):
        raise ValueError('Cada canal (e o Tempo) precisa apontar para uma coluna distinta do arquivo.')
    forca = [l for _, l, t, _ in p.canais if t == 'Força']
    if not canais_do_tipo(p, 'EMG'):
        raise ValueError('Cadastre ao menos um canal de EMG.')
    if not forca:
        raise ValueError('Cadastre ao menos um canal de força (célula de carga).')
    if len(forca) > 2 or (len(forca) == 2 and sorted(forca) != ['Dir', 'Esq']):
        raise ValueError('Use 1 canal de força (unilateral/único) ou 2 canais de força, um "Dir" e um "Esq".')


# ==============================================================================
# LEITURA E VALIDAÇÃO DO ARQUIVO
# ==============================================================================
def detecta_separador(linhas: list[str]) -> str:
    amostra = [l for l in linhas if l.strip()][:30]
    if not amostra:
        return 'tab'
    contadores = [('tab', lambda l: l.count('\t') + 1),
                  ('ponto_virgula', lambda l: l.count(';') + 1),
                  ('espaco', lambda l: len(l.split())),
                  ('virgula', lambda l: l.count(',') + 1)]
    for chave, conta in contadores:
        if np.median([conta(l) for l in amostra]) >= 2:
            return chave
    return 'tab'


def le_arquivo(conteudo: bytes, p: Parametros, avisos: list[str]) -> pd.DataFrame:
    texto = conteudo.decode('utf-8-sig', errors='replace')
    linhas = texto.splitlines()[p.linhas_cabecalho:]
    chave = p.separador if p.separador != 'auto' else detecta_separador(linhas)
    if chave not in SEPARADORES:
        raise ValueError(f'Separador desconhecido: {chave}')
    try:
        bruto = pd.read_csv(io.StringIO('\n'.join(linhas)), sep=SEPARADORES[chave],
                            header=None, decimal=p.decimal, skip_blank_lines=True)
    except pd.errors.EmptyDataError:
        raise ValueError('O arquivo está vazio (ou só tem cabeçalho).')
    except pd.errors.ParserError as e:
        raise ValueError(f'Não foi possível interpretar o arquivo com o separador "{chave}": {e}')
    bruto = bruto.dropna(axis=1, how='all')          # tabulação/separador sobrando no fim das linhas
    ncols = bruto.shape[1]
    mapa = [p.coluna_tempo] + [int(c) for _, _, _, c in p.canais]
    campos = ['Tempo'] + [f'{t}:{id_canal(n, l)}' for n, l, t, _ in p.canais]
    if max(mapa) > ncols:
        raise ValueError(f'O arquivo tem {ncols} coluna(s) (separador: {chave}), mas o mapeamento '
                         f'exige a coluna {max(mapa)}. Ajuste o separador, o número de linhas '
                         f'de cabeçalho ou o mapeamento de colunas.')
    sel = bruto.iloc[:, [i - 1 for i in mapa]].copy()
    sel.columns = campos
    df = sel.apply(pd.to_numeric, errors='coerce')
    ruins = df.isna().any(axis=1).to_numpy()
    if ruins.all():
        raise ValueError('Nenhuma linha numérica foi encontrada. Verifique o separador decimal, '
                         'as linhas de cabeçalho e o mapeamento de colunas.')
    if ruins.any():
        primeiras = (np.flatnonzero(ruins)[:5] + p.linhas_cabecalho + 1).tolist()
        avisos.append(f'{int(ruins.sum())} linha(s) com valores não numéricos foram removidas '
                      f'(primeiras, aprox.: {primeiras}). A remoção de linhas no meio do registro '
                      f'quebra a continuidade temporal.')
    return df.loc[~ruins].reset_index(drop=True)


def previa_arquivo(conteudo: bytes, p: Parametros, n_linhas: int = 8) -> pd.DataFrame:
    """Primeiras linhas do arquivo com as colunas numeradas a partir de 1 (para ajudar no mapeamento)."""
    texto = conteudo.decode('utf-8-sig', errors='replace')
    linhas = texto.splitlines()[p.linhas_cabecalho:p.linhas_cabecalho + n_linhas]
    chave = p.separador if p.separador != 'auto' else detecta_separador(linhas)
    df = pd.read_csv(io.StringIO('\n'.join(linhas)), sep=SEPARADORES[chave], header=None,
                     dtype=str, skip_blank_lines=True, engine='python').dropna(axis=1, how='all')
    df.columns = [f'Col. {i}' for i in range(1, df.shape[1] + 1)]
    return df


LIMIAR_BAIXA_FREQ_FORCA = 0.5    # força: >= 50% da potência abaixo de 10 Hz; EMG fica tipicamente < 10%


def perfil_sinal(x: np.ndarray, fs: float) -> dict:
    """Características que distinguem força (lenta, de um só sinal) de EMG (rápido, oscila em torno de zero)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    xc = x - x.mean()
    if len(x) < 64 or not np.any(xc):
        return {'baixa_freq': np.nan, 'media': float(np.mean(x)) if len(x) else np.nan,
                'min': float(np.min(x)) if len(x) else np.nan, 'max': float(np.max(x)) if len(x) else np.nan}
    f, pxx = welch(xc, fs, nperseg=min(len(xc), 4096))
    return {'baixa_freq': float(pxx[f < 10].sum() / pxx.sum()), 'media': float(x.mean()),
            'min': float(x.min()), 'max': float(x.max())}


def tipo_provavel(perfil: dict) -> str:
    bf = perfil['baixa_freq']
    if not np.isfinite(bf):
        return '?'
    if bf >= LIMIAR_BAIXA_FREQ_FORCA:
        return 'Força'
    if bf <= 0.2:
        return 'EMG'
    return '?'


def diagnostico_colunas(conteudo: bytes, p: Parametros) -> pd.DataFrame:
    """Uma linha por coluna do arquivo: % da potência abaixo de 10 Hz, faixa de valores e tipo provável."""
    texto = conteudo.decode('utf-8-sig', errors='replace')
    linhas = texto.splitlines()[p.linhas_cabecalho:]
    chave = p.separador if p.separador != 'auto' else detecta_separador(linhas[:30])
    corpo = '\n'.join(linhas)
    if p.decimal == ',' and chave != 'virgula':
        corpo = corpo.replace(',', '.')
    df = pd.read_csv(io.StringIO(corpo), sep=SEPARADORES[chave], header=None, engine='python')
    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    out = []
    for i, c in enumerate(df.columns, 1):
        x = df[c].to_numpy(float)
        d = np.diff(x[np.isfinite(x)])
        crescente = len(d) > 10 and np.all(d > 0)
        pf = perfil_sinal(x, p.fs)
        out.append({'Coluna': i, 'Tipo provável': 'Tempo' if crescente else tipo_provavel(pf),
                    '% potência < 10 Hz': 100 * pf['baixa_freq'] if not crescente else np.nan,
                    'Média': pf['media'], 'Mínimo': pf['min'], 'Máximo': pf['max']})
    return pd.DataFrame(out).set_index('Coluna')


def confere_mapeamento(diag: pd.DataFrame, p: Parametros) -> tuple[list[str], list[str]]:
    """Compara o tipo configurado de cada coluna com o tipo provável. Retorna (erros, sugestões)."""
    erros, sug = [], []
    if p.coluna_tempo in diag.index and diag.loc[p.coluna_tempo, 'Tipo provável'] != 'Tempo':
        erros.append(f'A coluna {p.coluna_tempo}, configurada como Tempo, não é crescente.')
    for nome, lado, tipo, col in p.canais:
        if col not in diag.index:
            continue
        prov = diag.loc[col, 'Tipo provável']
        bf = diag.loc[col, '% potência < 10 Hz']
        rot = nome_canal(id_canal(nome, lado))
        if tipo == 'Força' and prov != 'Força':
            erros.append(f'Coluna {col} ({rot}) está configurada como FORÇA, mas o sinal oscila rápido em torno de zero, '
                         f'como um EMG (só {bf:.0f}% da potência abaixo de 10 Hz; uma célula de carga costuma ter > 90%).')
        if tipo == 'EMG' and prov == 'Força':
            erros.append(f'Coluna {col} ({rot}) está configurada como EMG, mas o sinal é lento como uma força '
                         f'({bf:.0f}% da potência abaixo de 10 Hz).')
    if erros:
        cand = [int(c) for c in diag.index if diag.loc[c, 'Tipo provável'] == 'Força']
        if cand:
            sug.append(f'Colunas com aspecto de célula de carga: {", ".join(map(str, cand))}. Confira na prévia e ajuste '
                       f'"Colunas do registro".')
        else:
            sug.append('Nenhuma coluna do arquivo tem aspecto de célula de carga: confira se a força foi exportada neste '
                       'arquivo.')
    return erros, sug


def cabecalho_arquivo(conteudo: bytes, p: Parametros) -> list[str]:
    """Nomes da última linha de cabeçalho (se houver), para sugerir o mapeamento."""
    if p.linhas_cabecalho <= 0:
        return []
    texto = conteudo.decode('utf-8-sig', errors='replace').splitlines()
    if len(texto) < p.linhas_cabecalho:
        return []
    linha = texto[p.linhas_cabecalho - 1]
    chave = p.separador if p.separador != 'auto' else detecta_separador(texto[p.linhas_cabecalho:p.linhas_cabecalho + 30])
    return [c.strip() for c in re.split(SEPARADORES[chave], linha.strip()) if c.strip()]


def verifica_tempo(tempo: np.ndarray, fs: float, avisos: list[str]) -> dict:
    info = {'fs_estimada': None, 'unidade_tempo': None}
    d = np.diff(tempo)
    if d.size < 2:
        return info
    if np.any(d <= 0):
        avisos.append('A coluna Tempo não é estritamente crescente.')
    dt = float(np.median(d))
    if dt <= 0:
        return info
    cand = {'s': 1 / dt, 'ms': 1000 / dt}
    unidade = min(cand, key=lambda k: abs(cand[k] - fs))
    info.update(fs_estimada=cand[unidade], unidade_tempo=unidade)
    if abs(cand[unidade] - fs) / fs > 0.01:
        avisos.append(f'A coluna Tempo sugere ~{cand[unidade]:.1f} Hz (assumindo {unidade}), '
                      f'mas fs configurada = {fs:g} Hz. Confira o parâmetro de amostragem.')
    if np.any(np.abs(d - dt) > 0.1 * dt):
        avisos.append('Intervalos de tempo irregulares (lacunas ou perda de amostras).')
    return info


def _fracao_no_extremo(x: np.ndarray) -> float:
    y = np.abs(x - np.mean(x))
    ext = y.max()
    return float(np.mean(y >= 0.999 * ext)) if ext > 0 else 1.0


# ==============================================================================
# FILTROS
# ==============================================================================
def filtra_emg(sinal: np.ndarray, p: Parametros) -> np.ndarray:
    """Remove o offset, aplica notch (opcional) + passa-banda Butterworth em cascata SOS, fase zero."""
    x = np.asarray(sinal, float)
    x = x - x.mean()
    secoes = []
    if p.usar_notch:
        b, a = iirnotch(p.f_notch, p.q_notch, p.fs)
        secoes.append(tf2sos(b, a))
    secoes.append(butter(p.ordem_emg, list(p.banda_emg), btype='bandpass', fs=p.fs, output='sos'))
    return sosfiltfilt(np.vstack(secoes), x)


def filtra_forca(sinal: np.ndarray, p: Parametros) -> np.ndarray:
    sos = butter(p.ordem_forca, p.fc_forca, btype='low', fs=p.fs, output='sos')
    return sosfiltfilt(sos, np.asarray(sinal, float))


# ==============================================================================
# MECÂNICA
# ==============================================================================
def _media_movel_valida(x: np.ndarray, n: int) -> np.ndarray:
    cs = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    return (cs[n:] - cs[:-n]) / n


def estima_linha_base(forca: np.ndarray, fs: float, janela_s: float) -> tuple[float, float]:
    """Nível e desvio-padrão do trecho mais 'quieto' (menor média móvel) do registro."""
    n = max(2, min(int(round(janela_s * fs)), len(forca)))
    ini = int(np.argmin(_media_movel_valida(forca, n)))
    trecho = forca[ini:ini + n]
    return float(trecho.mean()), float(trecho.std())


def detecta_onset(forca: np.ndarray, idx_pico: int, ruido: float, pct: float, n_sd: float):
    """Onset = 1 amostra após a última vez em que a força esteve abaixo do limiar antes do pico.
    Limiar = max(pct * pico, n_sd * ruído de repouso)."""
    limiar = max(pct * forca[idx_pico], n_sd * ruido)
    abaixo = np.flatnonzero(forca[:idx_pico] < limiar)
    idx_onset = int(abaixo[-1] + 1) if abaixo.size else 0
    return idx_onset, float(limiar)


def analisa_mecanica(forca: np.ndarray, ruido: float, nome: str, p: Parametros, avisos: list[str]) -> dict:
    fs = p.fs
    idx_pico = int(np.argmax(forca))
    pico = float(forca[idx_pico])
    if pico <= 0:
        avisos.append(f'[{nome}] a força tarada não é positiva em nenhum ponto; verifique o sinal/tara.')
    idx_onset, limiar = detecta_onset(forca, idx_pico, ruido, p.onset_pct_pico, p.n_sd_ruido)
    if p.n_sd_ruido * ruido > p.onset_pct_pico * pico:
        avisos.append(f'[{nome}] o piso de ruído ({p.n_sd_ruido:g} DP = {p.n_sd_ruido * ruido:.3g}) excede '
                      f'{p.onset_pct_pico:.0%} do pico; o limiar de onset foi elevado para {limiar:.3g}.')
    if idx_onset == 0:
        avisos.append(f'[{nome}] a força já estava acima do limiar na primeira amostra do trecho analisado; '
                      f'o onset não é confiável (inclua um trecho de repouso antes da contração).')
    rfd = {}
    for w in p.janelas_rfd_ms:
        idx_fim = idx_onset + int(round(w / 1000 * fs))
        if idx_fim >= len(forca):
            rfd[w] = np.nan
            avisos.append(f'[{nome}] RFD 0–{w} ms não calculado: o registro termina antes dessa janela.')
            continue
        if idx_fim > idx_pico:
            avisos.append(f'[{nome}] o pico ocorre antes de {w} ms após o onset; RFD 0–{w} ms inclui dados pós-pico.')
        rfd[w] = float((forca[idx_fim] - forca[idx_onset]) / (w / 1000))
    return {'idx_pico': idx_pico, 'idx_onset': idx_onset, 'limiar': limiar, 'pico': pico,
            'tempo_onset_pico_ms': (idx_pico - idx_onset) / fs * 1000, 'rfd': rfd}


def localiza_janela_civm(forca_total: np.ndarray, n: int) -> tuple[int, int]:
    """Janela de n amostras com maior força média. Retorna (início, fim) com fim exclusivo."""
    ini = int(np.argmax(_media_movel_valida(forca_total, n)))
    return ini, ini + n


def assimetria(d, e) -> float:
    d, e = float(d), float(e)
    den = max(abs(d), abs(e))
    return 100 * (d - e) / den if den > 0 and np.isfinite(den) else float('nan')


# ==============================================================================
# RESULTADO E PIPELINE
# ==============================================================================
@dataclass
class Resultado:
    parametros: dict
    nome_arquivo: str
    sha256: str
    n_amostras: int
    duracao_s: float
    info_fs: dict
    unidade_forca: str
    unidade_emg: str
    seg: tuple                      # (i0, i1) do recorte, em amostras do registro
    t_seg: np.ndarray               # tempo absoluto (s) do trecho analisado
    forcas: dict                    # id do canal de força -> força filtrada e tarada (trecho)
    ruido: dict                     # id do canal de força -> DP do ruído de repouso
    mec: dict                       # id do canal de força -> saída de analisa_mecanica
    janela: tuple                   # (ini, fim) em amostras do trecho (fim exclusivo)
    janela_auto: bool
    emg_seg: dict                   # canal -> EMG filtrado (trecho)
    t_full: np.ndarray
    emg_bruto: dict
    emg_filt: dict
    psd: dict                       # canal -> (f, p_bruto, p_filtrado)
    atenuacao_notch: pd.Series
    df_mec: pd.DataFrame
    df_emg: pd.DataFrame
    df_emg_assim: pd.DataFrame
    avisos: list = field(default_factory=list)
    taras: dict = field(default_factory=dict)        # id do canal de força -> tara (unidade de saída)
    emg_ids: list = field(default_factory=list)      # ordem dos canais de EMG
    forca_ids: list = field(default_factory=list)    # ordem dos canais de força

    @property
    def janela_s(self) -> tuple:
        return float(self.t_seg[self.janela[0]]), float(self.t_seg[self.janela[1] - 1])


def processa(conteudo: bytes, nome_arquivo: str, p: Parametros) -> Resultado:
    valida_parametros(p)
    avisos: list[str] = []
    df = le_arquivo(conteudo, p, avisos)
    fs = p.fs
    n = len(df)
    if n < 200:
        raise ValueError(f'Registro muito curto ({n} amostras).')
    info_fs = verifica_tempo(df['Tempo'].to_numpy(), fs, avisos)
    t_full = np.arange(n) / fs

    # unidades
    if p.unidade_forca_entrada == p.unidade_forca_saida:
        fator_f = 1.0
    else:
        fator_f = KGF_PARA_N if p.unidade_forca_entrada == 'kgf' else 1 / KGF_PARA_N
    emg_ids = canais_do_tipo(p, 'EMG')
    forca_ids = canais_do_tipo(p, 'Força')
    emg_bruto = {c: df[f'EMG:{c}'].to_numpy(float) * p.fator_emg for c in emg_ids}
    carga_bruta = {f: df[f'Força:{f}'].to_numpy(float) * fator_f for f in forca_ids}

    for c, v in emg_bruto.items():
        if _fracao_no_extremo(v) > 0.002:
            avisos.append(f'[{nome_canal(c)}] muitas amostras no valor extremo: possível saturação (clipping).')

    # filtragem do registro inteiro (o recorte é aplicado depois, para não criar transientes nas bordas)
    emg_filt = {c: filtra_emg(v, p) for c, v in emg_bruto.items()}
    forca_filt = {f: filtra_forca(v, p) for f, v in carga_bruta.items()}

    # recorte
    i0 = int(round(p.recorte_ini_s * fs))
    i1 = n if p.recorte_fim_s <= 0 else min(n, int(round(p.recorte_fim_s * fs)))
    n_jan = int(round(p.janela_civm_s * fs))
    if i0 < 0 or i1 - i0 < n_jan:
        raise ValueError('O trecho analisado é menor que a janela de CIVM. Ajuste o recorte ou a duração da janela.')
    sl = slice(i0, i1)
    t_seg = t_full[sl]
    emg_seg = {c: v[sl] for c, v in emg_filt.items()}

    # tara (trecho mais quieto) e mecânica de cada canal de força
    forcas, ruido, taras = {}, {}, {}
    for f_id in forca_ids:
        f = forca_filt[f_id][sl]
        nivel, sd = estima_linha_base(f, fs, p.janela_base_s)
        forcas[f_id], ruido[f_id] = f - nivel, sd
        taras[f_id] = float(nivel)
    mec = {f_id: analisa_mecanica(forcas[f_id], ruido[f_id], nome_canal(f_id), p, avisos) for f_id in forca_ids}

    # janela de CIVM (soma dos canais de força)
    total = np.sum([forcas[f_id] for f_id in forca_ids], axis=0)
    if p.janela_manual_ini_s is None:
        ini, fim = localiza_janela_civm(total, n_jan)
        janela_auto = True
    else:
        ini = int(round(p.janela_manual_ini_s * fs)) - i0
        fim = ini + n_jan
        janela_auto = False
        if ini < 0 or fim > len(total):
            raise ValueError('A janela de CIVM definida manualmente cai fora do trecho analisado.')
    tol = int(TOL_PICO_JANELA_S * fs)
    for f_id in forca_ids:
        if not (ini - tol <= mec[f_id]['idx_pico'] < fim + tol):
            avisos.append(f'[{nome_canal(f_id)}] o pico de força (t = {t_seg[mec[f_id]["idx_pico"]]:.2f} s) está a mais de '
                          f'{TOL_PICO_JANELA_S:g} s da janela de CIVM. Confira se há mais de uma contração no registro.')

    # unidades de exibição
    uf, ue = p.unidade_forca_saida, p.unidade_emg
    fm = {f_id: float(forcas[f_id][ini:fim].mean()) for f_id in forca_ids}
    fm_total = float(total[ini:fim].mean())
    bilateral = len(forca_ids) == 2
    forca_por_lado = {lado_de(f_id): f_id for f_id in forca_ids if lado_de(f_id)}

    def forca_ref(canal_emg: str) -> tuple[float, str]:
        """Força usada na ENM do músculo: a do mesmo lado; senão, a força total."""
        lado = lado_de(canal_emg)
        if bilateral and lado in forca_por_lado:
            f_id = forca_por_lado[lado]
            return fm[f_id], nome_canal(f_id)
        return fm_total, 'total'

    # tabela de mecânica (linhas = métricas; colunas = canais de força [+ assimetria se D/E])
    cols = [rotulo_forca(p, f_id) for f_id in forca_ids]
    if bilateral:
        f_d, f_e = forca_por_lado['Dir'], forca_por_lado['Esq']
    met = {}

    def add(rotulo, chave, com_assim=True):
        vals = {rotulo_forca(p, f_id): chave(f_id) for f_id in forca_ids}
        if bilateral:
            vals['Assimetria D–E (%)'] = assimetria(chave(f_d), chave(f_e)) if com_assim else np.nan
        met[rotulo] = vals

    add(f'Pico de força ({uf})', lambda f: mec[f]['pico'])
    add(f'Força média na janela ({uf})', lambda f: fm[f])
    add('Instante do onset (s)', lambda f: t_seg[mec[f]['idx_onset']], False)
    add('Instante do pico (s)', lambda f: t_seg[mec[f]['idx_pico']], False)
    add('Tempo onset–pico (ms)', lambda f: mec[f]['tempo_onset_pico_ms'], False)
    for w in p.janelas_rfd_ms:
        add(f'RFD 0–{w} ms ({uf}/s)', lambda f, w=w: mec[f]['rfd'][w])
    df_mec = pd.DataFrame(met).T[cols + (['Assimetria D–E (%)'] if bilateral else [])]

    # tabela de EMG (uma linha por canal)
    linhas, rms, enm = {}, {}, {}
    for c in emg_ids:
        r = float(np.sqrt(np.mean(emg_seg[c][ini:fim] ** 2)))
        f_ref, origem = forca_ref(c)
        rms[c] = r
        enm[c] = f_ref / r if r > 0 else np.nan
        linhas[nome_canal(c)] = {f'RMS ({ue})': r,
                                 f'Força média da janela ({uf})': f_ref,
                                 f'Eficiência neuromuscular ({uf}/{ue})': enm[c],
                                 'Força de referência': origem}
    df_emg = pd.DataFrame(linhas).T

    # comparação bilateral: músculos que têm canal Dir e Esq
    linhas = {}
    for nome in dict.fromkeys(n for n, l, t, _ in p.canais if t == 'EMG'):
        cd, ce = id_canal(nome, 'Dir'), id_canal(nome, 'Esq')
        if cd in rms and ce in rms:
            linhas[nome] = {f'RMS Dir ({ue})': rms[cd], f'RMS Esq ({ue})': rms[ce],
                            'Assimetria RMS (%)': assimetria(rms[cd], rms[ce]),
                            'ENM Dir': enm[cd], 'ENM Esq': enm[ce],
                            'Assimetria ENM (%)': assimetria(enm[cd], enm[ce])}
    df_emg_assim = pd.DataFrame(linhas).T

    # auditoria espectral (registro inteiro)
    nps = min(1024, n)
    psd = {}
    for c in emg_ids:
        f_, pb = welch(emg_bruto[c] - emg_bruto[c].mean(), fs, nperseg=nps)
        _, pf = welch(emg_filt[c], fs, nperseg=nps)
        psd[c] = (f_, pb, pf)
    atenu = {}
    if 0 < p.f_notch < fs / 2:
        for c, (f_, pb, pf) in psd.items():
            banda = np.abs(f_ - p.f_notch) <= 2
            if banda.any() and pf[banda].sum() > 0 and pb[banda].sum() > 0:
                atenu[nome_canal(c)] = 10 * np.log10(pb[banda].sum() / pf[banda].sum())
    atenuacao_notch = pd.Series(atenu, name=f'Atenuação em {p.f_notch:g} ± 2 Hz (dB)', dtype=float)

    from dataclasses import asdict
    return Resultado(
        parametros=asdict(p), nome_arquivo=nome_arquivo, sha256=hashlib.sha256(conteudo).hexdigest(),
        n_amostras=n, duracao_s=n / fs, info_fs=info_fs, unidade_forca=uf, unidade_emg=ue,
        seg=(i0, i1), t_seg=t_seg, forcas=forcas, ruido=ruido, mec=mec, janela=(ini, fim),
        janela_auto=janela_auto, emg_seg=emg_seg, t_full=t_full, emg_bruto=emg_bruto, emg_filt=emg_filt,
        psd=psd, atenuacao_notch=atenuacao_notch, df_mec=df_mec, df_emg=df_emg,
        df_emg_assim=df_emg_assim, avisos=avisos,
        emg_ids=emg_ids, forca_ids=forca_ids, taras=taras)


# ==============================================================================
# FORMATAÇÃO (pt-BR) E METADADOS
# ==============================================================================
def fmt(x, casas: int = 2) -> str:
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return '—'
    if isinstance(x, str):
        return x
    s = f'{float(x):,.{casas}f}'
    return s.replace(',', '§').replace('.', ',').replace('§', '.')


def casas_padrao(linha, coluna) -> int:
    l, c = str(linha), str(coluna)
    if '(%)' in c or '(%)' in l:
        return 1
    if '% da' in l or '% do' in l or '% da' in c:
        return 1
    if 'Eficiência' in c or 'ENM' in c or 'Eficiência' in l or 'ENM' in l:
        return 4                                  # kgf/µV ~ 0,04: precisa de 4 casas
    if 'RMS' in l or 'RMS' in c:
        return 1 if ('µV' in c or 'µV' in l or 'uV' in c or 'uV' in l) else 4
    if '(s)' in l:
        return 3
    if '(ms)' in l:
        return 1
    return 2


def formata_tabela(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
    for i in df.index:
        for c in df.columns:
            out.loc[i, c] = fmt(df.loc[i, c], casas_padrao(i, c))
    return out


def descricao_parametros(p: Parametros) -> list[tuple[str, str, str]]:
    notch = f'{p.f_notch:g} Hz, Q = {p.q_notch:g}' if p.usar_notch else 'desativado'
    modo = 'automática (maior força média da soma dos canais de força)' if p.janela_manual_ini_s is None \
        else f'manual (início em {p.janela_manual_ini_s:g} s)'
    fim = 'até o final' if p.recorte_fim_s <= 0 else f'{p.recorte_fim_s:g} s'
    canais = '; '.join(f'{nome_canal(id_canal(n, l))} ({t}, col. {c})' for n, l, t, c in p.canais)
    return [
        ('Aquisição', 'Canais', f'Tempo (col. {p.coluna_tempo}); {canais}'),
        ('Aquisição', 'Frequência de amostragem', f'{p.fs:g} Hz'),
        ('Aquisição', 'Recorte analisado', f'{p.recorte_ini_s:g} s a {fim}'),
        ('Aquisição', 'Unidade de força (entrada → saída)', f'{p.unidade_forca_entrada} → {p.unidade_forca_saida}'),
        ('Aquisição', 'Unidade / fator de escala do EMG', f'{p.unidade_emg} / × {p.fator_emg:g}'),
        ('EMG', 'Passa-banda Butterworth (fase zero)',
         f'{p.banda_emg[0]:g}–{p.banda_emg[1]:g} Hz, protótipo de ordem {p.ordem_emg}, dupla passagem (filtfilt)'),
        ('EMG', 'Notch (fase zero)', notch),
        ('Força', 'Passa-baixa Butterworth (fase zero)', f'{p.fc_forca:g} Hz, ordem {p.ordem_forca}, dupla passagem (filtfilt)'),
        ('Força', 'Linha de base (tara)', f'janela de {p.janela_base_s:g} s com menor média móvel'),
        ('Onset', 'Limiar', f'máx({p.onset_pct_pico:.1%} do pico; {p.n_sd_ruido:g} DP do ruído de repouso)'),
        ('Onset', 'Busca', 'de trás para frente a partir do pico'),
        ('CIVM', 'Duração da janela', f'{p.janela_civm_s:g} s'),
        ('CIVM', 'Posição da janela', modo),
        ('RFD', 'Janelas a partir do onset', ', '.join(f'{w} ms' for w in p.janelas_rfd_ms)),
        ('Auditoria', 'PSD de Welch', 'janela de Hann, 1024 amostras (ou o registro, se menor), 50% de sobreposição'),
    ]


def versoes() -> dict:
    import matplotlib
    v = {'Python': platform.python_version(), 'NumPy': np.__version__,
         'SciPy': __import__('scipy').__version__, 'pandas': pd.__version__,
         'Matplotlib': matplotlib.__version__, 'Aplicação': APP_VERSION}
    try:
        import weasyprint
        v['WeasyPrint'] = weasyprint.__version__
    except Exception:
        pass
    return v
