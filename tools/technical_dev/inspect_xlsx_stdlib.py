"""Print workbook sheets and sample rows using Python's standard library."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import PurePosixPath

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "p": "http://schemas.openxmlformats.org/package/2006/relationships"}


def read_workbook(path: str):
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        shared = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.findall(".//m:t", NS))
                      for si in root.findall("m:si", NS)]
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        relroot = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rels = {r.attrib["Id"]: r.attrib["Target"] for r in relroot}
        result = []
        for sheet in wb.findall("m:sheets/m:sheet", NS):
            target = rels[sheet.attrib[f"{{{NS['r']}}}id"]]
            member = str(PurePosixPath("xl") / target)
            if member not in names:
                member = str(PurePosixPath("xl") / "worksheets" / PurePosixPath(target).name)
            root = ET.fromstring(zf.read(member))
            table = []
            for row in root.findall(".//m:sheetData/m:row", NS):
                cells = []
                for cell in row.findall("m:c", NS):
                    ref = cell.attrib.get("r", "A1")
                    letters = "".join(ch for ch in ref if ch.isalpha())
                    col = 0
                    for ch in letters:
                        col = col * 26 + ord(ch.upper()) - 64
                    col -= 1
                    value = cell.find("m:v", NS)
                    text = value.text if value is not None else ""
                    if cell.attrib.get("t") == "s" and text:
                        text = shared[int(text)]
                    elif cell.attrib.get("t") == "inlineStr":
                        text = "".join(t.text or "" for t in cell.findall(".//m:t", NS))
                    cells.append((col, text))
                if cells:
                    rowv = [""] * (max(i for i, _ in cells) + 1)
                    for i, value in cells:
                        rowv[i] = value
                    table.append(rowv)
            result.append((sheet.attrib["name"], table))
        return result


if __name__ == "__main__":
    for workbook in sys.argv[1:]:
        print("WORKBOOK", workbook)
        for name, rows in read_workbook(workbook):
            print("SHEET", name, "rows", len(rows))
            for row in rows[:7]:
                print(repr(row))
