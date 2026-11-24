"""Relatório estatístico em PDF (HTML + CSS -> WeasyPrint)."""
import base64
import html
import io
from datetime import datetime

import matplotlib.pyplot as plt

import nucleo as nc
from graficos import fig_forca, fig_onset, fig_emg, fig_psd

CSS = """
@page { size: A4; margin: 17mm 15mm 18mm 15mm;
  @bottom-left  { content: "Relatório de análise EMG + força — CIVM"; font-size: 7.5pt; color: #777; }
  @bottom-right { content: "Página " counter(page) " de " counter(pages); font-size: 7.5pt; color: #777; } }
body { font-family: "DejaVu Sans", "Helvetica", "Arial", sans-serif; font-size: 8.8pt; color: #1b1f24; line-height: 1.38; }
h1 { font-size: 17pt; margin: 0 0 2pt 0; color: #0f2a4a; }
.sub { color: #55606d; font-size: 9pt; margin-bottom: 10pt; }
h2 { font-size: 11.5pt; color: #0f2a4a; border-bottom: 1.2pt solid #0f2a4a; padding-bottom: 2pt; margin: 15pt 0 6pt 0; page-break-after: avoid; }
h3 { font-size: 9.5pt; margin: 9pt 0 3pt 0; color: #22364f; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 7pt 0; page-break-inside: avoid; }
th, td { border: 0.5pt solid #c5ccd6; padding: 2.4pt 5pt; font-size: 8pt; }
th { background: #eaf0f7; text-align: center; font-weight: bold; }
td { text-align: right; }
td:first-child, th:first-child { text-align: left; }
table.txt td { text-align: left; }
table.txt td:first-child { width: 34%; font-weight: bold; background: #f6f8fb; }
figure { margin: 5pt 0 8pt 0; page-break-inside: avoid; }
figure img { width: 100%; }
figcaption { font-size: 7.6pt; color: #55606d; }
.aviso { background: #fff6e0; border-left: 3pt solid #e0a000; padding: 3pt 7pt; margin: 2pt 0; font-size: 8pt; }
.ok { background: #eaf7ee; border-left: 3pt solid #2f9e5b; padding: 3pt 7pt; font-size: 8pt; }
.mono { font-family: "DejaVu Sans Mono", monospace; font-size: 7pt; word-break: break-all; }
.nota { font-size: 7.6pt; color: #55606d; }
"""


def _b64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _tabela(df, classe='', cabecalho_indice=' '):
    fmt_df = nc.formata_tabela(df)
    fmt_df.insert(0, cabecalho_indice, [html.escape(str(i)) for i in df.index])
    fmt_df.columns = [html.escape(str(c)) for c in fmt_df.columns]
    return fmt_df.to_html(escape=False, index=False, border=0, classes=classe)


def _tabela_txt(linhas):
    corpo = ''.join(f'<tr><td>{html.escape(str(a))}</td><td>{html.escape(str(b))}</td></tr>' for a, b in linhas)
    return f'<table class="txt">{corpo}</table>'


def _figura(fig, legenda):
    return f'<figure><img src="data:image/png;base64,{_b64(fig)}"><figcaption>{html.escape(legenda)}</figcaption></figure>'


def monta_html(res, meta: dict, extras: dict | None = None) -> str:
    extras = extras or {}
    p = nc.Parametros(**res.parametros)
    uf = res.unidade_forca
    agora = datetime.now().strftime('%d/%m/%Y %H:%M')
    ini_s, fim_s = res.janela_s
    modo = 'automática' if res.janela_auto else 'manual'

    ident = [('Arquivo analisado', res.nome_arquivo),
             ('Participante / ID', meta.get('participante') or '—'),
             ('Avaliador', meta.get('avaliador') or '—'),
             ('Protocolo / posição', meta.get('protocolo') or '—'),
             ('Observações', meta.get('observacoes') or '—'),
             ('Duração do registro', f'{nc.fmt(res.duracao_s, 2)} s ({res.n_amostras} amostras a {p.fs:g} Hz)'),
             ('Trecho analisado', f'{nc.fmt(res.t_seg[0], 2)} s a {nc.fmt(res.t_seg[-1], 2)} s'),
             ('Janela de CIVM', f'{nc.fmt(ini_s, 3)} s a {nc.fmt(ini_s + p.janela_civm_s, 3)} s '
                                f'({p.janela_civm_s:g} s, {modo})')]

    if res.avisos:
        avisos_html = ''.join(f'<div class="aviso">{html.escape(a)}</div>' for a in res.avisos)
    else:
        avisos_html = '<div class="ok">Nenhum aviso de qualidade foi gerado nas verificações automáticas.</div>'

    if len(res.atenuacao_notch):
        atenu = res.atenuacao_notch.to_frame()
        tab_atenu = _tabela(atenu, cabecalho_indice='Canal')
    else:
        tab_atenu = ''

    if len(res.df_emg_assim):
        bilateral_html = ('<h3>Comparação bilateral por músculo</h3>'
                          + _tabela(res.df_emg_assim, cabecalho_indice='Músculo'))
    else:
        bilateral_html = ''
    if 'Assimetria D–E (%)' in res.df_mec.columns:
        nota_assim = ('Assimetria D–E (%) = 100 × (D − E) / máx(|D|, |E|); valores positivos indicam o lado '
                      'direito maior.\n')
    else:
        nota_assim = ''

    vers = _tabela_txt(list(nc.versoes().items()))
    params = _tabela_txt([(f'{g} — {n}', v) for g, n, v in nc.descricao_parametros(p)])

    return f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Relatório CIVM</title><style>{CSS}</style></head><body>
<h1>Relatório de análise EMG + força na CIVM</h1>
<div class="sub">Gerado em {agora} — versão da aplicação {nc.APP_VERSION}</div>

<h2>1. Identificação</h2>
{_tabela_txt(ident)}

<h2>2. Mecânica (células de carga)</h2>
{_tabela(res.df_mec, cabecalho_indice='Variável')}
<p class="nota">{nota_assim}Onset = 1 amostra após a última vez em que a força esteve abaixo do limiar antes do pico. RFD = ΔForça/Δt entre o onset e o
instante indicado.</p>
{_figura(fig_forca(res), f'Figura 1. Força tarada ({uf}) de cada canal, com onset, pico, limiar de onset e janela de CIVM.')}
{_figura(fig_onset(res), 'Figura 2. Ampliação em torno do onset, com os instantes usados no cálculo do RFD (quadrados).')}
{extras.get('mec', '')}

<h2>3. Eletromiografia</h2>
{_tabela(res.df_emg, cabecalho_indice='Músculo')}
{bilateral_html}
<p class="nota">RMS calculado sobre o EMG filtrado, na mesma janela de CIVM da força. Eficiência neuromuscular (ENM) =
força média da janela ({uf}) do lado correspondente (ou a força total, indicada na coluna "Força de referência") ÷ RMS
do músculo; como a força é a do membro e não de cada músculo,
interpretar a ENM como contribuição relativa, não como eficiência intrínseca do músculo.</p>
{_figura(fig_emg(res), 'Figura 3. EMG filtrado (trecho analisado); a faixa verde é a janela de CIVM.')}
{extras.get('emg', '')}

<h2>4. Controle de qualidade</h2>
{avisos_html}
<h3>Auditoria espectral</h3>
{tab_atenu}
{_figura(fig_psd(res), 'Figura 4. Densidade espectral de potência (Welch) do sinal bruto (sem offset) e filtrado; linha vermelha = frequência do notch.')}

<h2>5. Parâmetros de processamento e rastreabilidade</h2>
{params}
<h3>Integridade do arquivo de entrada (SHA-256)</h3>
<div class="mono">{res.sha256}</div>
<h3>Versões de software</h3>
{vers}
<p class="nota">Relatório gerado automaticamente. Onset, pico e janela de CIVM devem ser conferidos visualmente
nas Figuras 1 e 2 antes de qualquer interpretação.</p>
{extras.get('glossario', '')}
{extras.get('resumo', '')}
</body></html>"""


def html_para_pdf(doc: str) -> bytes:
    """HTML -> PDF. Tenta WeasyPrint; se faltar, usa Chrome/Chromium/Edge headless instalado; senão, RuntimeError."""
    erros = []
    try:
        from weasyprint import HTML
        return HTML(string=doc).write_pdf()
    except Exception as e:  # noqa: BLE001
        erros.append(f'WeasyPrint: {e}')
    import os
    import shutil
    import subprocess
    import tempfile
    for exe in ('google-chrome', 'chromium', 'chromium-browser', 'chrome', 'msedge'):
        caminho = shutil.which(exe)
        if not caminho:
            continue
        with tempfile.TemporaryDirectory() as d:
            h, pdf = os.path.join(d, 'r.html'), os.path.join(d, 'r.pdf')
            with open(h, 'w', encoding='utf-8') as f:
                f.write(doc)
            subprocess.run([caminho, '--headless=new', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer',
                            f'--print-to-pdf={pdf}', h], capture_output=True, timeout=180)
            if os.path.exists(pdf) and os.path.getsize(pdf) > 0:
                with open(pdf, 'rb') as f:
                    return f.read()
        erros.append(f'{exe}: falhou')
    raise RuntimeError('Não foi possível gerar o PDF: instale o WeasyPrint (no Streamlit Cloud, mantenha o '
                       'packages.txt) ou tenha o Chrome/Chromium instalado. ' + ' | '.join(erros))


def gera_pdf(res, meta: dict, extras: dict | None = None) -> bytes:
    return html_para_pdf(monta_html(res, meta, extras))


def extras_completos(res, pe) -> dict:
    """TDF, cocontração, assimetrias e glossário para o relatório (+ avisos das análises extras)."""
    import avancado as av
    import glossario as gl
    ex = av.calcula(res, pe)
    blocos = av.secoes_html(res, ex, pe, _tabela, _figura)
    blocos['glossario'] = ('<h2 style="page-break-before:always">6. Glossário: o que cada variável significa e como foi '
                           'calculada</h2>' + gl.html_glossario(nc.Parametros(**res.parametros), pe))
    import resumo as rs
    blocos['resumo'] = rs.html_resumo(res, ex, pe, _figura)
    blocos['avisos'] = ex.avisos
    return blocos
