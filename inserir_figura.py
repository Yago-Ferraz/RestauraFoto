"""
Insere a Figura 1 no RestauraFoto_Relatorio.docx,
logo após a legenda "Figura 1 – Fluxo de processamento do RestauraFoto."
"""

from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path
import shutil

DOCX_IN  = "RestauraFoto_Relatorio.docx"
DOCX_OUT = "RestauraFoto_Relatorio.docx"
FIG_PATH = "figura1_arquitetura.png"

if not Path(FIG_PATH).exists():
    print(f"ERRO: {FIG_PATH} não encontrado. Execute gerar_figura_arquitetura.py primeiro.")
    exit(1)

shutil.copy(DOCX_IN, DOCX_IN + ".bak")

doc = Document(DOCX_IN)

# Localiza o parágrafo da legenda e insere a figura antes dele
TARGET = "Figura 1"
for i, para in enumerate(doc.paragraphs):
    if TARGET in para.text and "Fluxo de processamento" in para.text:
        # Insere a imagem no parágrafo ANTERIOR (que deve estar vazio ou ser o texto)
        # Na API python-docx não há insert_before nativo, então usamos XML
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        import copy

        # Cria um novo parágrafo com a imagem
        new_para = OxmlElement("w:p")
        new_run  = OxmlElement("w:r")

        # Adiciona a imagem via método interno
        temp_para = doc.add_paragraph()
        temp_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = temp_para.add_run()
        run.add_picture(FIG_PATH, width=Inches(5.8))

        # Move o elemento XML para antes da legenda
        temp_xml = temp_para._element
        para._element.addprevious(copy.deepcopy(temp_xml))
        temp_para._element.getparent().remove(temp_para._element)

        print(f"Figura inserida antes do parágrafo {i}: '{para.text[:50]}'")
        break
else:
    print("Legenda 'Figura 1' não encontrada — adicionando ao final da seção 3.1")
    # Fallback: adiciona após o primeiro parágrafo que menciona "Figura 1"
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(FIG_PATH, width=Inches(5.8))

doc.save(DOCX_OUT)
print(f"Salvo em {DOCX_OUT}  (backup em {DOCX_IN}.bak)")
