"""Generate a large test PDF for benchmarking."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_benchmark_pdf(output_path: Path, num_pages: int = 40) -> None:
    """Generate a multi-page PDF with text paragraphs and tables."""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    for page_num in range(num_pages):
        # Add header
        elements.append(Paragraph(f"Page {page_num + 1}", styles["Heading1"]))

        # Add text paragraphs
        for para_num in range(3):
            text = (
                f"This is paragraph {para_num + 1} on page {page_num + 1}. "
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
                "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
            )
            elements.append(Paragraph(text, styles["Normal"]))
            elements.append(Spacer(1, 12))

        # Add a table every few pages
        if page_num % 3 == 0:
            data = [
                ["ID", "Name", "Value", "Status"],
                *[[f"{i}", f"Item {i}", f"${i * 10}.00", "Active"] for i in range(1, 6)],
            ]
            table = Table(data)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            elements.append(table)
            elements.append(Spacer(1, 24))

        # Add page break after each page except the last
        if page_num < num_pages - 1:
            elements.append(PageBreak())

    doc.build(elements)


if __name__ == "__main__":
    output = Path(__file__).parent / "data" / "benchmark_large.pdf"
    output.parent.mkdir(exist_ok=True)
    generate_benchmark_pdf(output, num_pages=40)
    print(f"Generated: {output}")
