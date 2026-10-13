"""Post-process OMML in a docx: | and ‖ delimiters, hide empty n-ary upper limits, Cambria Math font on math runs."""
import re, sys, zipfile, shutil
src = sys.argv[1]; tmp = src + ".tmp"
zin = zipfile.ZipFile(src)
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        x = data.decode("utf-8")
        x = x.replace('<m:begChr m:val="["/><m:endChr m:val="]"/>', '<m:begChr m:val="|"/><m:endChr m:val="|"/>')
        x = x.replace('<m:begChr m:val="{"/><m:endChr m:val="}"/>', '<m:begChr m:val="‖"/><m:endChr m:val="‖"/>')
        # empty upper limit on a sum -> hide it (otherwise Word shows an empty placeholder box)
        x = re.sub(r'(<m:naryPr>(?:(?!</m:naryPr>).)*?)(</m:naryPr>)(<m:sub>(?:(?!</m:sub>).)*</m:sub><m:sup/>)',
                   r'\1<m:supHide m:val="1"/>\2\3', x)
        x = x.replace('<m:r><m:t', '<m:r><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr><m:t')
        data = x.encode("utf-8")
    zout.writestr(item, data)
zout.close(); zin.close(); shutil.move(tmp, src)
print("fixed", src)
