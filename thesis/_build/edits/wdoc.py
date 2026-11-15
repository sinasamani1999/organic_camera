"""Small in-place editor for the thesis Word files (keeps the user's own formatting).

Text helpers accept a light markup: **bold**, $tex$ (inline equation, converted to OMML).
Persian 'digit-digit' sequences get the same RLM treatment as the original build scripts.
"""
import copy
import re
import shutil
import zipfile

from lxml import etree

from omml import omml

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
NS = {"w": W, "m": M}
q = lambda tag: "{%s}%s" % (W, tag)
RLM = "‏"


def fa_dash(text):
    return re.sub(r"([۰-۹])-(?=[۰-۹\s])", "\\1" + RLM + "-" + RLM, text)


def ptext(p):
    return "".join(t.text or "" for t in p.iter(q("t"), "{%s}t" % M)).replace(RLM, "")


class Doc:
    def __init__(self, path):
        self.path = path
        self.z = zipfile.ZipFile(path)
        self.root = etree.fromstring(self.z.read("word/document.xml"))
        self.body = self.root.find(q("body"))

    # ------------------------------------------------------------ find
    def paras(self):
        return list(self.body.iter(q("p")))

    def find(self, sub, nth=0, start=None):
        sub = sub.replace(RLM, "")
        hits = [p for p in self.paras() if sub in ptext(p)]
        if start is not None:
            order = {id(p): i for i, p in enumerate(self.paras())}
            hits = [p for p in hits if order[id(p)] > order[id(start)]]
        if len(hits) <= nth:
            raise KeyError("paragraph not found: " + sub[:60])
        return hits[nth]

    def table_after(self, caption_sub):
        cap = self.find(caption_sub)
        el = cap.getnext()
        while el is not None and el.tag != q("tbl"):
            el = el.getnext()
        return el

    # ------------------------------------------------------------ write
    def _run_rpr(self, p):
        r = p.find(".//" + q("r"))
        rpr = r.find(q("rPr")) if r is not None else None
        return copy.deepcopy(rpr) if rpr is not None else etree.Element(q("rPr"))

    def set_text(self, p, text, rtl=True, rpr=None):
        """Replace all content of paragraph p (keeps pPr). Supports **bold** and $tex$."""
        base = rpr if rpr is not None else self._run_rpr(p)
        for b in base.findall(q("b")) + base.findall(q("bCs")) + base.findall(q("i")) + base.findall(q("iCs")):
            base.remove(b)
        bold_default = rpr is not None and False
        for child in list(p):
            if child.tag != q("pPr"):
                p.remove(child)
        if rtl:
            text = fa_dash(text)
        parts = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\$[^$]+\$)", text)
        maths = [s[1:-1] for s in parts if s.startswith("$")]
        mx = iter(omml(maths)) if maths else iter(())
        for s in parts:
            if not s:
                continue
            if s.startswith("$"):
                p.append(etree.fromstring('<wrap xmlns:w="%s" xmlns:m="%s">%s</wrap>' % (W, M, next(mx)))[0])
                continue
            bold = s.startswith("**")
            ital = not bold and s.startswith("*") and s.endswith("*") and len(s) > 2
            if bold:
                s = s[2:-2]
            elif ital:
                s = s[1:-1]
            r = etree.SubElement(p, q("r"))
            rp = copy.deepcopy(base)
            if bold:
                rp.insert(1 if rp.find(q("rFonts")) is not None else 0, etree.Element(q("b")))
                rp.insert(2 if rp.find(q("rFonts")) is not None else 1, etree.Element(q("bCs")))
            if ital:
                sz = rp.find(q("sz"))
                (sz.addprevious if sz is not None else rp.append)(etree.Element(q("i")))
                sz = rp.find(q("sz"))
                (sz.addprevious if sz is not None else rp.append)(etree.Element(q("iCs")))
            if rtl and rp.find(q("rtl")) is None:
                rp.append(etree.Element(q("rtl")))
            if not rtl:
                for e in rp.findall(q("rtl")):
                    rp.remove(e)
            r.append(rp)
            t = etree.SubElement(r, q("t"))
            t.text = s
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        return p

    def clone(self, template):
        n = copy.deepcopy(template)
        for k in list(n.attrib):
            if k.startswith("{%s}" % W14):
                del n.attrib[k]
        return n

    def insert_after(self, anchor, template, text, rtl=True):
        n = self.set_text(self.clone(template), text, rtl=rtl, rpr=self._run_rpr(template))
        anchor.addnext(n)
        return n

    def insert_before(self, anchor, template, text, rtl=True):
        n = self.set_text(self.clone(template), text, rtl=rtl, rpr=self._run_rpr(template))
        anchor.addprevious(n)
        return n

    def add_reference(self, heading_sub, text):
        """Insert a reference paragraph in alphabetical order under the references heading."""
        h = self.find(heading_sub)
        refs, el = [], h.getnext()
        while el is not None and el.tag == q("p"):
            if ptext(el).strip():
                refs.append(el)
            el = el.getnext()
        key = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower().replace("é", "e").replace("ä", "a"))
        new_key = key(re.sub(r"\*", "", text))
        for r in refs:
            if key(ptext(r)) > new_key:
                return self.insert_before(r, r, text, rtl=False)
        return self.insert_after(refs[-1], refs[-1], text, rtl=False)

    def remove(self, p):
        p.getparent().remove(p)

    # ------------------------------------------------------------ tables
    def rows(self, tbl):
        return tbl.findall(q("tr"))

    def set_cell(self, row, i, text):
        tc = row.findall(q("tc"))[i]
        ps = tc.findall(q("p"))
        for extra in ps[1:]:
            tc.remove(extra)
        rtl = bool(re.search(r"[؀-ۿ]", text))
        self.set_text(ps[0], text, rtl=rtl)

    def add_row_after(self, row, cells):
        n = self.clone(row)
        row.addnext(n)
        for i, c in enumerate(cells):
            self.set_cell(n, i, c)
        return n

    # ------------------------------------------------------------ equations
    def equation_after(self, anchor, template_eq, tex):
        """Clone a display-equation paragraph (label runs + tab + oMath) and put a new formula in it."""
        n = self.clone(template_eq)
        old = n.find(".//{%s}oMath" % M)
        new = etree.fromstring('<wrap xmlns:w="%s" xmlns:m="%s">%s</wrap>' % (W, M, omml([tex])[0]))[0]
        old.addprevious(new)
        n.remove(old) if old.getparent() is n else old.getparent().remove(old)
        anchor.addnext(n)
        return n

    def renumber_equations(self, ch):
        """Sequential numbers for display equations whose label is '(ch' '‑' 'NN)'."""
        fa = lambda k: str(k).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))
        k = 0
        for p in self.paras():
            if p.find(".//{%s}oMath" % M) is None:
                continue
            ts = [t for t in p.iter(q("t"))]
            if len(ts) >= 3 and ts[0].text == "(" + ch and ts[1].text == "‑" and re.fullmatch(r"[۰-۹]+\)", ts[2].text or ""):
                k += 1
                ts[2].text = fa(k) + ")"
        return k

    # ------------------------------------------------------------ save
    def save(self, out):
        data = etree.tostring(self.root, xml_declaration=True, encoding="UTF-8", standalone=True)
        tmp = out + ".tmp"
        with zipfile.ZipFile(self.path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                zout.writestr(item, data if item.filename == "word/document.xml" else zin.read(item.filename))
        shutil.move(tmp, out)
