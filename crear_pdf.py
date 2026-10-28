from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=14)
pdf.multi_cell(0, 10, "Documento de prueba para el laboratorio RAG\n\n")
pdf.set_font("Helvetica", size=11)

texto = (
    "La empresa TechNova S.L. fue fundada en 2018 en Valencia, Espana. "
    "Su actividad principal es el desarrollo de software de inteligencia artificial "
    "para el sector sanitario. En 2023 facturo 4,2 millones de euros y cuenta con "
    "45 empleados. Su producto estrella se llama MediAssist, un asistente virtual "
    "que ayuda a los medicos a redactar informes clinicos. "
    "El director ejecutivo es Marta Ruiz y la directora tecnica es Lucia Gomez. "
    "La empresa tiene previsto abrir oficina en Lisboa en 2027."
)

pdf.multi_cell(0, 8, texto)
pdf.output("documento_prueba.pdf")
print("PDF creado: documento_prueba.pdf")
