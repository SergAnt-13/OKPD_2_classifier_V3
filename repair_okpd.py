# repair_okpd.py (исправленный)
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

source = Path("data/reference/okpd_2.xlsx")
output = Path("data/reference/okpd_2_repaired.xlsx")

with zipfile.ZipFile(source, 'r') as zf:
    shared_strings_xml = zf.read('xl/sharedStrings.xml')
    sheet_xml = zf.read('xl/worksheets/sheet1.xml')

# Определяем пространство имён из корневого элемента
ns = {'s': 'http://purl.oclc.org/ooxml/spreadsheetml/main'}

# Парсим shared strings
root = ET.fromstring(shared_strings_xml)
shared_strings = [si.findtext('s:t', '', ns) for si in root.findall('s:si', ns)]

# Парсим sheet1
sheet_root = ET.fromstring(sheet_xml)
rows_data = []
for row in sheet_root.findall('s:sheetData/s:row', ns):
    cells = []
    for c in row.findall('s:c', ns):
        value_elem = c.find('s:v', ns)
        if value_elem is None:
            cells.append('')
        else:
            t = c.get('t')
            if t == 's':
                idx = int(value_elem.text)
                cells.append(shared_strings[idx] if idx < len(shared_strings) else '')
            else:
                cells.append(value_elem.text or '')
    rows_data.append(cells)

if rows_data:
    df = pd.DataFrame(rows_data[1:], columns=rows_data[0])
    df.to_excel(output, index=False)
    print(f"Восстановлено {len(df)} строк. Сохранено в {output}")
    print(f"Колонки: {list(df.columns)}")
else:
    print("Не удалось извлечь данные")