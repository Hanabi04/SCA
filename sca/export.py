"""Portable CSV and Office Open XML workbook export using the standard library."""

import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED


def column(n):
    result = ""
    while n:
        n, r = divmod(n - 1, 26)
        result = chr(65 + r) + result
    return result


def export_tables(directory, tables):
    """Write numeric XLSX cells and UTF-8-with-BOM CSVs for Excel."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    pkg = "http://schemas.openxmlformats.org/package/2006/relationships"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    for name, rows in tables.items():
        with (directory / f"{name}.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            csv.writer(f).writerows(rows)
    with ZipFile(directory / "reproduction_results.xlsx", "w", ZIP_DEFLATED) as z:
        overrides = "".join(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for i in range(1, len(tables) + 1)
        )
        z.writestr(
            "[Content_Types].xml",
            f'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>{overrides}</Types>',
        )
        z.writestr(
            "_rels/.rels",
            f'<Relationships xmlns="{pkg}"><Relationship Id="rId1" Type="{rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        )
        sheets = "".join(
            f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
            for i, name in enumerate(tables, 1)
        )
        z.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>{sheets}</sheets></workbook>',
        )
        links = "".join(
            f'<Relationship Id="rId{i}" Type="{rel}/worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(tables) + 1)
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{pkg}">{links}<Relationship Id="styles" Type="{rel}/styles" Target="styles.xml"/></Relationships>',
        )
        z.writestr(
            "xl/styles.xml",
            f"""<styleSheet xmlns="{ns}"><numFmts count="3"><numFmt numFmtId="164" formatCode="0.00"/><numFmt numFmtId="165" formatCode="0.00E+00"/><numFmt numFmtId="166" formatCode="0.0000"/></numFmts><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF203864"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="5"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"><alignment wrapText="1" vertical="center"/></xf><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"><alignment wrapText="1" vertical="center"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="166" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>""",
        )
        for i, (name, rows) in enumerate(tables.items(), 1):
            widths = [
                min(
                    48,
                    max(14, max(len(str(r[c])) if c < len(r) else 0 for r in rows) + 2),
                )
                for c in range(len(rows[0]))
            ]
            cols = "".join(
                f'<col min="{c}" max="{c}" width="{w}" customWidth="1"/>'
                for c, w in enumerate(widths, 1)
            )
            xmlrows = []
            for ri, row in enumerate(rows, 1):
                cells = []
                for ci, value in enumerate(row, 1):
                    if value is None:
                        continue
                    address = f"{column(ci)}{ri}"
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        style = (
                            0
                            if isinstance(value, int)
                            else (
                                3
                                if rows[0][ci - 1] == "McNemar p"
                                else (
                                    4 if rows[0][ci - 1] in ("Value", "IQR (s)") else 2
                                )
                            )
                        )
                        cells.append(f'<c r="{address}" s="{style}"><v>{value}</v></c>')
                    else:
                        cells.append(
                            f'<c r="{address}" s="{1 if ri==1 else 0}" t="inlineStr"><is><t xml:space="preserve">{escape(str(value))}</t></is></c>'
                        )
                height = max(
                    21,
                    max(
                        15 * math.ceil(len(str(v)) / max(1, widths[c] - 2))
                        for c, v in enumerate(row)
                        if v is not None
                    ),
                )
                xmlrows.append(
                    f'<row r="{ri}" ht="{32 if ri==1 else height}" customHeight="1">'
                    + "".join(cells)
                    + "</row>"
                )
            z.writestr(
                f"xl/worksheets/sheet{i}.xml",
                f'<worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols>{cols}</cols><sheetData>{"".join(xmlrows)}</sheetData><autoFilter ref="A1:{column(len(rows[0]))}{len(rows)}"/></worksheet>',
            )
