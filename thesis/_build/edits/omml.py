"""TeX (small subset, same parser as build3.js) -> OMML strings styled like the user's chapters."""
import json, re, subprocess, zipfile, os, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
RPR = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:cs="IRZar"/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>'

def fix(x):
    x = x.replace('<m:begChr m:val="["/><m:endChr m:val="]"/>', '<m:begChr m:val="|"/><m:endChr m:val="|"/>')
    x = x.replace('<m:begChr m:val="{"/><m:endChr m:val="}"/>', '<m:begChr m:val="‖"/><m:endChr m:val="‖"/>')
    x = re.sub(r'(<m:naryPr>(?:(?!</m:naryPr>).)*?)(</m:naryPr>)(<m:sub>(?:(?!</m:sub>).)*</m:sub><m:sup/>)', r'\1<m:supHide m:val="1"/>\2\3', x)
    x = re.sub(r'<m:r>(?:<w:rPr>.*?</w:rPr>)?<m:t', '<m:r>' + RPR + '<m:t', x)
    return x

def omml(texs):
    with tempfile.TemporaryDirectory() as d:
        j, o = os.path.join(d, "in.json"), os.path.join(d, "out.docx")
        json.dump(texs, open(j, "w", encoding="utf-8"))
        subprocess.run(["node", os.path.join(HERE, "eqgen.js"), j, o], check=True,
                       env={**os.environ, "NODE_PATH": "/opt/npm-tools/node_modules"})
        x = zipfile.ZipFile(o).read("word/document.xml").decode("utf-8")
    out = re.findall(r'<m:oMath>.*?</m:oMath>', x, flags=re.S)
    assert len(out) == len(texs), (len(out), len(texs))
    return [fix(m) for m in out]

if __name__ == "__main__":
    for m in omml(["L = \\alpha L_{pos} + \\delta L_{spec}", "\\hat{P}(f) = \\frac{1}{B}\\sum_{i=1}^{B}|X_i(f)|^2"]): print(m[:300], '\n')
