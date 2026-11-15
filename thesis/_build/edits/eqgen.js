const fs = require("fs");
const D = require("docx");
const { Document, Packer, Paragraph, Math: OMath, MathRun, MathSquareBrackets, MathCurlyBrackets, MathSum, MathFraction, MathRadical, MathSubScript, MathSuperScript, MathSubSuperScript, MathIntegral } = D;
// ---------------- tiny LaTeX subset -> OMML
const SYM = { "\\Delta": "Δ", "\\theta": "θ", "\\omega": "ω", "\\sigma": "σ", "\\odot": "⊙", "\\cdot": "·",
  "\\tanh": "tanh", "\\,": " ", "\\{": "{", "\\}": "}", "\\ldots": "…", "\\ddot{x}": "ẍ", "\\dot{x}": "ẋ",
  "\\tilde{h}": "h̃", "\\phi": "φ", "\\tilde{m}": "m̃", "\\hat{\\mathbf{y}}": "ŷ", "\\hat{y}": "ŷ",
  "\\delta": "δ", "\\rho": "ρ", "\\lambda": "λ", "\\tau": "τ", "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\mu": "μ",
  "\\quad": "   ", "\\in": "∈", "\\|": "‖", "\\cos": "cos", "\\arg": "arg ", "\\max": "max", "\\times": "×", "\\hat{P}": "P̂", "\\hat{\\mathbf{r}}": "r̂", "\\hat{r}": "r̂", "\\log": "log", "\\epsilon": "ε", "\\leftarrow": "←", "\\mathcal{F}": "ℱ" };
function parse(src) {
  let i = 0;
  function group() { // after '{'
    let depth = 1, j = i;
    while (depth) { if (src[j] === "{") depth++; else if (src[j] === "}") depth--; j++; }
    const inner = src.slice(i, j - 1); i = j; return inner;
  }
  function atom() { // one item for sub/sup
    if (src[i] === "{") { i++; return parse(group()); }
    if (src[i] === "\\") { const m = src.slice(i).match(/^\\[a-zA-Z]+/); i += m[0].length; return [new MathRun(SYM[m[0]] || m[0].slice(1))]; }
    return [new MathRun(src[i++])];
  }
  const out = [];
  let buf = "";
  const flush = () => { if (buf) { out.push(new MathRun(buf)); buf = ""; } };
  const scripts = base => {
    let sub = null, sup = null;
    while (src[i] === "_" || src[i] === "^") { const c = src[i++]; const a = atom(); if (c === "_") sub = a; else sup = a; }
    if (sub && sup) return new MathSubSuperScript({ children: base, subScript: sub, superScript: sup });
    if (sub) return new MathSubScript({ children: base, subScript: sub });
    if (sup) return new MathSuperScript({ children: base, superScript: sup });
    return null;
  };
  while (i < src.length) {
    const c = src[i];
    if (c === "|" || src.startsWith("\\|", i)) {
      const tok = c === "|" ? "|" : "\\|";
      const j = src.indexOf(tok, i + tok.length);
      const inner = parse(src.slice(i + tok.length, j)); i = j + tok.length; flush();
      const base = [tok === "|" ? new MathSquareBrackets({ children: inner }) : new MathCurlyBrackets({ children: inner })];
      const s2 = scripts(base); out.push(s2 || base[0]); continue;
    }
    if (c === "\\") {
      const long = Object.keys(SYM).filter(k => k.includes("{")).find(k => src.startsWith(k, i));
      let base;
      if (long) { i += long.length; base = [new MathRun(SYM[long])]; }
      else {
        const m = src.slice(i).match(/^\\([a-zA-Z]+|.)/); i += m[0].length; const cmd = m[1];
        if (cmd === "frac") { i++; const n = parse(group()); i++; const d = parse(group()); flush(); out.push(new MathFraction({ numerator: n, denominator: d })); continue; }
        if (cmd === "sqrt") { i++; const a = parse(group()); flush(); out.push(new MathRadical({ children: a })); continue; }
        if (cmd === "int") {
          let sub = [], sup = [];
          while (src[i] === "_" || src[i] === "^") { const k = src[i++]; const a = atom(); if (k === "_") sub = a; else sup = a; }
          flush(); out.push(new MathIntegral({ children: parse(src.slice(i)), subScript: sub, superScript: sup })); i = src.length; continue;
        }
        if (cmd === "mathrm") { i++; buf += group(); continue; }
        if (cmd === "sum") {
          let sub = [], sup = [];
          while (src[i] === "_" || src[i] === "^") { const k = src[i++]; const a = atom(); if (k === "_") sub = a; else sup = a; }
          flush(); out.push(new MathSum({ children: parse(src.slice(i)), subScript: sub, superScript: sup })); i = src.length; continue;
        }
        if (cmd === "mathbf" || cmd === "boldsymbol") {
          i++; const inner = parse(group()); flush();
          const s2 = scripts(inner); if (s2) out.push(s2); else out.push(...inner); continue;
        }
        base = [new MathRun(SYM["\\" + cmd] || cmd)];
      }
      flush();
      const s = scripts(base);
      out.push(s || base[0]);
      continue;
    }
    if (c === "_" || c === "^") { // scripts on the last character in buf
      const last = buf.slice(-1); buf = buf.slice(0, -1); flush();
      out.push(scripts([new MathRun(last)])); continue;
    }
    buf += c; i++;
  }
  flush();
  return out;
}
const mathInline = tex => new OMath({ children: parse(tex) });


const list = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const doc = new Document({ sections: [{ children: list.map(t => new Paragraph({ children: [new OMath({ children: parse(t) })] })) }] });
Packer.toBuffer(doc).then(b => fs.writeFileSync(process.argv[3], b));
