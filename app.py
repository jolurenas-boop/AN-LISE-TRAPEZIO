"""Aplicação Streamlit — TRAPÉZIO (1 eletrodo de EMG) + 1 célula de carga.

Analisa a CIVM (força, onset, pico, janela de CIVM, TDF 0–50/100/200 ms, RMS, ENM) e o teste de fadiga a 50% da
CIVM (patamar, MDF/MNF/RMS/ENM por janela de 0,5 s com 50% de sobreposição, artefatos), integrando os dois:
força da fadiga em % da CIVM e RMS da fadiga em % da RMS da CIVM. Gera um relatório único em PDF.

Arquivos (um por teste), colunas padrão: 1 Tempo | 2 Trapézio | 3 Célula de carga  (editável).
Executar:  streamlit run app.py
"""
import re
from dataclasses import asdict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import avancado as av
import coleta_unica as cu
import fadiga_individual as fi
import fadiga_nucleo as fn
import integra as ig
import nucleo as nc
from fadiga_graficos import fig_emg as f_emg_fad, fig_forca as f_forca_fad, fig_tendencia
from graficos import fig_emg as f_emg_civm, fig_forca as f_forca_civm, fig_onset, fig_psd
from relatorio import html_para_pdf

st.set_page_config(page_title='Trapézio — CIVM e fadiga', layout='wide')

SEP = {'Automático': 'auto', 'Tabulação': 'tab', 'Ponto e vírgula': 'ponto_virgula', 'Espaço(s)': 'espaco',
       'Vírgula': 'virgula'}
LADOS = {'Não especificado': '', 'Direito': 'Dir', 'Esquerdo': 'Esq'}
OPCOES_RFD = [25, 50, 75, 100, 150, 200, 250, 300]


def mostra(fig):
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)


def csv_ptbr(df, index=True) -> bytes:
    return ('﻿' + df.to_csv(sep=';', decimal=',', index=index)).encode('utf-8')


@st.cache_data(show_spinner=False, max_entries=16)
def diag_cache(conteudo: bytes, params: dict):
    return nc.diagnostico_colunas(conteudo, nc.Parametros(**params))


@st.cache_data(show_spinner='Processando a CIVM…', max_entries=16)
def civm_cache(conteudo: bytes, nome: str, params: dict):
    return nc.processa(conteudo, nome, nc.Parametros(**params))


@st.cache_data(show_spinner='Processando a fadiga…', max_entries=16)
def fadiga_cache(conteudo: bytes, nome: str, id_vol: str, colunas: dict, mapa: dict, fmt_arq: tuple, params: dict,
                 manual: dict, taras: dict):
    teste, _ = cu.monta_teste(conteudo, nome, id_vol, colunas['TEMPO'], {'TRP': colunas['TRP'], 'CC': colunas['CC']},
                              *fmt_arq)
    return cu.processa(teste, mapa, fn.Parametros(**params), manual, None, taras)


ss = st.session_state
ss.setdefault('manual', {})

# ==============================================================================
# BARRA LATERAL
# ==============================================================================
with st.sidebar:
    st.header('Arquivos')
    arq_civm = st.file_uploader('CIVM (.txt)', type=['txt', 'csv', 'tsv', 'dat'], key='civm')
    arq_fad = st.file_uploader(f'Fadiga a 50% da CIVM (.txt) — opcional', type=['txt', 'csv', 'tsv', 'dat'], key='fad')
    nome_ref = (arq_civm or arq_fad).name if (arq_civm or arq_fad) else ''
    _m = re.match(r'\s*(\d+)(?=[_\s-])', nome_ref)
    id_vol = st.text_input('Voluntário / ID', value=_m.group(1).zfill(3) if _m else '')
    with st.expander('Formato e colunas', expanded=False):
        sep = st.selectbox('Separador de colunas', list(SEP))
        dec = st.selectbox('Separador decimal', ['.', ','])
        n_cab = st.number_input('Linhas de cabeçalho a ignorar', 0, 1000, 0, 1)
        c_t = st.number_input('Coluna do Tempo', 1, 200, 1, 1)
        c_e = st.number_input('Coluna do Trapézio (EMG)', 1, 200, 2, 1)
        c_f = st.number_input('Coluna da Célula de carga', 1, 200, 3, 1)
        lado_rot = st.selectbox('Lado do trapézio / da célula', list(LADOS))
        st.caption('As mesmas colunas valem para os dois arquivos. O app confere se a coluna de força tem mesmo sinal '
                   'de força e a de EMG sinal de EMG.')

    st.header('Parâmetros')
    with st.expander('Aquisição e filtros'):
        fs = st.number_input('Frequência de amostragem (Hz)', 100.0, 50000.0, 2000.0, 100.0)
        u_f = st.selectbox('Unidade da força (no arquivo e nos resultados)', ['kgf', 'N'])
        u_emg = st.text_input('Unidade do EMG', 'µV')
        f_emg = st.number_input('Fator de escala do EMG', 0.0, 1e6, 1.0, format='%g')
        b_lo = st.number_input('EMG passa-alta (Hz)', 1.0, 200.0, 20.0, 1.0)
        b_hi = st.number_input('EMG passa-baixa (Hz)', 50.0, 2000.0, 450.0, 10.0)
        notch = st.checkbox('Notch', True)
        f_notch = st.number_input('Frequência do notch (Hz)', 1.0, 1000.0, 60.0, 1.0)
        fc_f = st.number_input('Força: passa-baixa (Hz)', 1.0, 100.0, 20.0, 1.0)
    with st.expander('CIVM: onset, janela e TDF'):
        j_base = st.number_input('Janela da linha de base (s)', 0.05, 5.0, 0.5, 0.05)
        pct = st.number_input('Limiar de onset (% do pico)', 0.1, 50.0, 2.0, 0.5)
        n_sd = st.number_input('Piso de ruído (nº de DP)', 0.0, 20.0, 3.0, 0.5)
        j_civm = st.number_input('Duração da janela de CIVM (s)', 0.1, 10.0, 1.0, 0.1)
        manual = st.checkbox('Janela de CIVM manual', False)
        j_ini = st.number_input('Início da janela manual (s)', 0.0, 600.0, 0.0, 0.1, disabled=not manual)
        rfd = st.multiselect('Janelas de TDF (ms após o onset)', OPCOES_RFD, default=[50, 100, 200])
    with st.expander('Fadiga: alvo e patamar', expanded=True):
        alvo = st.number_input('Alvo (% da CIVM)', 5.0, 100.0, 50.0, 5.0)
        tol = st.number_input('Tolerância do alvo (± % da CIVM)', 1.0, 30.0, 5.0, 1.0)
        k_dp = st.number_input('Patamar: média ± k·DP — k =', 0.5, 5.0, 2.0, 0.25)
        dur_min = st.number_input('Permanecer na faixa por (s)', 0.25, 10.0, 2.0, 0.25)
        cv_max = st.number_input('Verificação: CV máximo (%)', 1.0, 50.0, 10.0, 1.0)
        der_max = st.number_input('Verificação: deriva máxima (%)', 1.0, 100.0, 20.0, 1.0)
        reserva = st.number_input('Se não estabilizar: onset + (s)', 0.0, 20.0, 3.0, 0.5)
        fim_ign = st.number_input('Descartar o fim do registro (s)', 0.0, 10.0, 0.0, 0.25)
    with st.expander('Fadiga: janelas e artefatos'):
        jan = st.number_input('Janela espectral (s)', 0.1, 5.0, 0.5, 0.05)
        sob = st.number_input('Sobreposição (%)', 0, 90, 50, 5)
        seg = st.number_input('Segmento do Welch (s)', 0.05, 5.0, 0.25, 0.05)
        ext = st.number_input('Inicial/final: primeiros/últimos (s)', 0.25, 10.0, 2.0, 0.25)
        excl = st.checkbox('Excluir janelas com artefato', True)
        fe = st.number_input('Limite do conversor (± unidade do EMG)', 0.0, 1e6, 5000.0, 100.0)
        pico_dp = st.number_input('Pico máximo (nº de DP robustos)', 3.0, 100.0, 15.0, 1.0)
        rmsr = st.number_input('RMS máximo (× mediana)', 1.2, 20.0, 2.5, 0.1)
    with st.expander('Identificação no relatório'):
        avaliador = st.text_input('Avaliador')
        observ = st.text_area('Observações', height=70)

lado = LADOS[lado_rot]
canais_civm = (('Trapézio', lado, 'EMG', int(c_e)), ('Célula de carga', lado, 'Força', int(c_f)))
mapa_fad = {'TRP': ('Trapézio', lado, 'EMG'), 'CC': ('Célula de carga', lado, 'Força')}
try:
    p_civm = nc.Parametros(
        fs=float(fs), separador=SEP[sep], decimal=dec, linhas_cabecalho=int(n_cab), coluna_tempo=int(c_t),
        canais=canais_civm, unidade_forca_entrada=u_f, unidade_forca_saida=u_f, unidade_emg=u_emg or 'µV',
        fator_emg=float(f_emg), banda_emg=(float(b_lo), float(b_hi)), usar_notch=bool(notch), f_notch=float(f_notch),
        fc_forca=float(fc_f), janela_civm_s=float(j_civm), janela_manual_ini_s=float(j_ini) if manual else None,
        janela_base_s=float(j_base), onset_pct_pico=float(pct) / 100, n_sd_ruido=float(n_sd),
        janelas_rfd_ms=tuple(sorted(rfd)))
    nc.valida_parametros(p_civm)
    p_fad = fn.Parametros(
        fs=float(fs), unidade_emg=u_emg or 'µV', unidade_forca=u_f, banda_emg=(float(b_lo), float(b_hi)),
        usar_notch=bool(notch), f_notch=float(f_notch), fc_forca=float(fc_f), k_dp=float(k_dp),
        dur_min_estavel_s=float(dur_min), cv_max_pct=float(cv_max), deriva_max_pct=float(der_max),
        inicio_reserva_s=float(reserva), fim_ignorado_s=float(fim_ign), janela_espectral_s=float(jan),
        sobreposicao=float(sob) / 100, welch_segmento_s=float(seg), extremos_s=float(ext), excluir_artefatos=bool(excl),
        fundo_escala=float(fe), pico_max_dp=float(pico_dp), rms_max_razao=float(rmsr))
    fn.valida_parametros(p_fad)
except Exception as e:  # noqa: BLE001
    st.error(f'Configuração inválida: {e}')
    st.stop()
PT = ig.ParametrosTeste(alvo_pct=float(alvo), tolerancia_pct=float(tol))
PE = av.ParametrosExtra(agonista='Trapézio', antagonista='—', tarefa='CIVM')

# ==============================================================================
# CORPO
# ==============================================================================
st.title('Trapézio — CIVM e fadiga a 50% da CIVM')
if arq_civm is None:
    st.info('Envie o arquivo da CIVM (e, se tiver, o da fadiga a 50%) na barra lateral.')
    st.markdown(f'**Colunas esperadas:** `Col. {c_t}: Tempo | Col. {c_e}: Trapézio (EMG) | Col. {c_f}: Célula de carga`')
    st.markdown('A CIVM define o alvo da fadiga (50% do pico) e normaliza o EMG da fadiga (% da RMS da CIVM). '
                'Sem a CIVM, a fadiga não pode ser expressa em % da CIVM.')
    st.stop()


def confere(conteudo, nome):
    """Verificação das colunas; para o app se o mapeamento não conferir (a menos que o usuário force)."""
    try:
        d = diag_cache(conteudo, asdict(p_civm))
        erros, sug = nc.confere_mapeamento(d, p_civm)
    except Exception as e:  # noqa: BLE001
        st.warning(f'{nome}: não foi possível verificar as colunas ({e}).')
        return []
    if not erros:
        return []
    st.error(f'**{nome}: as colunas configuradas não conferem com o conteúdo do arquivo.** Os resultados sairiam errados.')
    for m in erros + sug:
        st.warning(m)
    st.dataframe(nc.formata_tabela(d))
    if not st.checkbox(f'Ignorar a verificação de {nome} (não recomendado)', False, key=f'ign_{nome}'):
        st.stop()
    return [f'VERIFICAÇÃO DE COLUNAS IGNORADA ({nome}): ' + m for m in erros]


ign = confere(arq_civm.getvalue(), 'CIVM')
try:
    r_civm = civm_cache(arq_civm.getvalue(), arq_civm.name, asdict(p_civm))
except ValueError as e:
    st.error(f'CIVM: {e}')
    st.stop()
r_fad = None
if arq_fad is not None:
    ign += confere(arq_fad.getvalue(), 'Fadiga')
    f_id = r_civm.forca_ids[0]
    taras = {'CC': (r_civm.taras[f_id], f'CIVM ({arq_civm.name})')}
    try:
        r_fad = fadiga_cache(arq_fad.getvalue(), arq_fad.name, id_vol or 'sem_id',
                             {'TEMPO': int(c_t), 'TRP': int(c_e), 'CC': int(c_f)}, mapa_fad, (SEP[sep], dec, int(n_cab)),
                             asdict(p_fad), ss['manual'], taras)
    except Exception as e:  # noqa: BLE001
        st.error(f'Fadiga: {e}')

ex = av.calcula(r_civm, PE)
integ, jan_norm, av_int = ig.integra(r_civm, r_fad, PT) if r_fad is not None else (None, None, [])
avisos = ign + list(r_civm.avisos) + ([f'[Fadiga] {a}' for a in r_fad.avisos] if r_fad is not None else []) + av_int
if avisos:
    with st.expander(f'Avisos de qualidade ({len(avisos)})', expanded=True):
        for a in avisos:
            st.warning(a)

abas = ['Resumo visual', 'CIVM', 'TDF'] + (['Fadiga', 'CIVM → fadiga'] if r_fad is not None else []) + \
       ['Relatório e exportação', 'Glossário', 'Parâmetros']
T = dict(zip(abas, st.tabs(abas)))

with T['Resumo visual']:
    for d in ig.destaques(r_civm, ex, r_fad, integ, PT):
        st.markdown('- ' + d.replace('<b>', '**').replace('</b>', '**'))
    mostra(ig.fig_resumo(r_civm, ex, r_fad, integ, jan_norm, PT))

with T['CIVM']:
    st.dataframe(nc.formata_tabela(r_civm.df_mec))
    st.dataframe(nc.formata_tabela(r_civm.df_emg.drop(columns=[c for c in r_civm.df_emg.columns if 'referência' in c])))
    mostra(f_forca_civm(r_civm))
    mostra(f_emg_civm(r_civm))
    with st.expander('Auditoria espectral'):
        mostra(fig_psd(r_civm))

with T['TDF']:
    st.dataframe(nc.formata_tabela(ex.tdf))
    mostra(av.fig_tdf(ex.tdf, r_civm.unidade_forca))
    mostra(fig_onset(r_civm))

if r_fad is not None:
    with T['Fadiga']:
        lado_f = next(iter(r_fad.forca))
        st.dataframe(nc.formata_tabela(fi.tabela_mecanica([r_fad], lado_f)))
        mostra(f_forca_fad(r_fad))
        with st.expander('Ajustar o início do patamar à mão'):
            v = st.text_input('Início do patamar (s)', value=f'{ss["manual"][lado_f]:g}' if lado_f in ss['manual'] else '',
                              placeholder=f'auto: {r_fad.forca[lado_f]["i_ini"] / p_fad.fs:.2f}')
            if st.button('Aplicar'):
                try:
                    ss['manual'] = {lado_f: float(v.replace(',', '.'))} if v.strip() else {}
                    st.rerun()
                except ValueError:
                    st.error('Número inválido')
        st.dataframe(nc.formata_tabela(r_fad.df_emg))
        st.dataframe(nc.formata_tabela(r_fad.df_emg_extra))
        t1, t2, t3, t4 = st.tabs(['MDF', 'RMS', 'ENM', 'EMG filtrado'])
        with t1:
            mostra(fig_tendencia(r_fad, 'MDF_Hz', 'MDF (Hz)'))
        with t2:
            mostra(fig_tendencia(r_fad, 'RMS', f'RMS ({p_fad.unidade_emg})'))
        with t3:
            mostra(fig_tendencia(r_fad, 'ENM', f'ENM ({u_f}/{p_fad.unidade_emg})'))
        with t4:
            mostra(f_emg_fad(r_fad))
    with T['CIVM → fadiga']:
        st.dataframe(nc.formata_tabela(integ))
        st.caption('Força da fadiga em % do pico da CIVM; RMS da fadiga em % da RMS do trapézio na janela de CIVM; '
                   'ENM da fadiga em % da ENM da CIVM.')

with T['Relatório e exportação']:
    meta = {'participante': id_vol, 'musculo': f'Trapézio {lado_rot.lower()}' if lado else 'Trapézio',
            'avaliador': avaliador, 'observacoes': ' '.join(filter(None, [observ] + ign))}
    nomes = {'civm': arq_civm.name, 'fadiga': arq_fad.name if arq_fad else None}
    stem = re.sub(r'[^A-Za-z0-9_-]+', '_', id_vol or arq_civm.name.rsplit('.', 1)[0])
    nome_pdf = f'{stem}_trapezio_CIVM' + ('_fadiga50' if r_fad is not None else '') + '.pdf'
    if st.button('Gerar relatório em PDF', type='primary'):
        with st.spinner('Montando o relatório…'):
            doc = ig.monta_html(r_civm, r_fad, PT, PE, meta, nomes)
            try:
                ss['pdf'] = (nome_pdf, html_para_pdf(doc), 'application/pdf')
            except RuntimeError as e:
                st.warning(f'{e} — entregando em HTML (abra e use Imprimir → Salvar como PDF).')
                ss['pdf'] = (nome_pdf.replace('.pdf', '.html'), doc.encode('utf-8'), 'text/html')
    if ss.get('pdf'):
        st.download_button(f'Baixar {ss["pdf"][0]}', ss['pdf'][1], ss['pdf'][0], ss['pdf'][2])
        st.caption('Se mudar arquivos ou parâmetros, gere o relatório novamente.')
    c1, c2, c3 = st.columns(3)
    c1.download_button('CIVM — mecânica e TDF (CSV)', csv_ptbr(pd.concat([r_civm.df_mec, ex.tdf])), f'{stem}_civm.csv')
    if r_fad is not None:
        c2.download_button('Fadiga — EMG e força (CSV)',
                           csv_ptbr(pd.concat([r_fad.df_emg, r_fad.df_emg_extra], axis=1)), f'{stem}_fadiga_emg.csv')
        c3.download_button('Fadiga — janelas (CSV)', csv_ptbr(jan_norm, False), f'{stem}_fadiga_janelas.csv')

with T['Glossário']:
    st.dataframe(ig.linhas_glossario(p_civm, PE, p_fad if r_fad is not None else None, PT), hide_index=True)

with T['Parâmetros']:
    st.table([{'Teste': 'CIVM', 'Grupo': g, 'Parâmetro': n, 'Valor': v} for g, n, v in nc.descricao_parametros(p_civm)]
             + ([{'Teste': 'Fadiga', 'Grupo': g, 'Parâmetro': n, 'Valor': v} for g, n, v in fn.descricao_parametros(p_fad)]
                if r_fad is not None else []))
    st.json(nc.versoes())
