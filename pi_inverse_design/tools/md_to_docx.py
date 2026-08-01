import argparse
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


NS_RE = re.compile(r"^(.*?<w:body>)", re.S)
TAIL_RE = re.compile(r"(<w:sectPr[\s\S]*?</w:sectPr>\s*</w:body>\s*</w:document>)$", re.S)


def parse_blocks(text: str):
    blocks = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("# "):
            blocks.append(("title", line[2:].strip()))
        elif line.startswith("## "):
            blocks.append(("h1", line[3:].strip()))
        elif line.startswith("### "):
            blocks.append(("h2", line[4:].strip()))
        elif line.startswith("- "):
            blocks.append(("bullet", line[2:].strip()))
        else:
            blocks.append(("p", line.strip()))
    return blocks


def run_xml(text: str, bold=False, size=24):
    text = escape(text)
    bold_xml = "<w:b/>" if bold else ""
    return (
        "<w:r>"
        "<w:rPr>"
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
        'w:eastAsia="SimSun" w:cs="Times New Roman"/>'
        f"{bold_xml}"
        f'<w:sz w:val="{size}"/>'
        f'<w:szCs w:val="{size}"/>'
        "</w:rPr>"
        f'<w:t xml:space="preserve">{text}</w:t>'
        "</w:r>"
    )


def paragraph_xml(kind: str, text: str):
    if kind == "title":
        ppr = (
            "<w:pPr>"
            '<w:spacing w:before="120" w:after="200" w:line="360" w:lineRule="auto"/>'
            '<w:jc w:val="center"/>'
            "</w:pPr>"
        )
        return f"<w:p>{ppr}{run_xml(text, bold=True, size=32)}</w:p>"

    if kind == "h1":
        ppr = (
            "<w:pPr>"
            '<w:spacing w:before="120" w:after="120" w:line="320" w:lineRule="auto"/>'
            "</w:pPr>"
        )
        return f"<w:p>{ppr}{run_xml(text, bold=True, size=28)}</w:p>"

    if kind == "h2":
        ppr = (
            "<w:pPr>"
            '<w:spacing w:before="80" w:after="80" w:line="300" w:lineRule="auto"/>'
            "</w:pPr>"
        )
        return f"<w:p>{ppr}{run_xml(text, bold=True, size=24)}</w:p>"

    if kind == "bullet":
        ppr = (
            "<w:pPr>"
            '<w:spacing w:before="0" w:after="40" w:line="300" w:lineRule="auto"/>'
            '<w:ind w:left="420"/>'
            "</w:pPr>"
        )
        return f"<w:p>{ppr}{run_xml('• ' + text, bold=False, size=24)}</w:p>"

    ppr = (
        "<w:pPr>"
        '<w:spacing w:before="0" w:after="40" w:line="320" w:lineRule="auto"/>'
        '<w:ind w:firstLine="420"/>'
        "</w:pPr>"
    )
    return f"<w:p>{ppr}{run_xml(text, bold=False, size=24)}</w:p>"


def build_document_xml(template_xml: str, markdown_text: str):
    m1 = NS_RE.search(template_xml)
    m2 = TAIL_RE.search(template_xml)
    if not m1 or not m2:
        raise RuntimeError("cannottemplate DOCX  document.xml structure")

    prefix = m1.group(1)
    suffix = m2.group(1)
    body = "".join(paragraph_xml(kind, text) for kind, text in parse_blocks(markdown_text))
    return prefix + body + suffix


def convert(md_path: Path, template_docx: Path, out_docx: Path):
    markdown_text = md_path.read_text(encoding="utf-8")
    with zipfile.ZipFile(template_docx, "r") as zin:
        template_xml = zin.read("word/document.xml").decode("utf-8")
        new_xml = build_document_xml(template_xml, markdown_text)

        with zipfile.ZipFile(out_docx, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = new_xml.encode("utf-8")
                zout.writestr(item, data)


def main():
    parser = argparse.ArgumentParser(description="Convert simple Markdown to DOCX with a template")
    parser.add_argument("--input", required=True, help="Markdown input file")
    parser.add_argument("--template", required=True, help="Template DOCX file")
    parser.add_argument("--output", required=True, help="Output DOCX file")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.template), Path(args.output))


if __name__ == "__main__":
    main()
