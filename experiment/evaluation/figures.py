"""Regenerate the report figures from sweep.json into docs/figures/.

Two metrics, and only two: AIM's own workload error, and range query error
stratified by interval width.  Four arms, a 2x2 over privacy definition
(bounded / unbounded) and release (stock / policy).

Colour encodes the privacy definition, dash encodes the arm, so the 2x2 reads
off the legend directly.

    python experiment/evaluation/figures.py     # from the project root
"""
import json

import numpy as np

RES = json.load(open("experiment/results/sweep.json"))
OUT = "docs/figures/"

RHOS = sorted({r["rho"] for r in RES})
BANDS = ["1-2", "3-5", "6-10", "11-20", ">20"]
ORD = {"0": ("age", 7), "1": ("hours.per.week", 8), "2": ("education.num", 2)}

# (bounded, policy) -> (label, colour, dashed)
ARMS = [(True, False, "bounded stock", "s1", True),
        (True, True, "bounded policy", "s1", False),
        (False, False, "unbounded stock", "s2", True),
        (False, True, "unbounded policy", "s2", False)]

W, H = 680, 380
L, RGT, TOP, BOT = 66, 150, 66, 52          # margins: plot box is what's left

CSS = (":root{--sf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;"
       "--gr:#e1e0d9;--ax:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;}"
       "@media (prefers-color-scheme:dark){:root{--sf:#1a1a19;--ink:#ffffff;"
       "--ink2:#c3c2b7;--mut:#898781;--gr:#2c2c2a;--ax:#383835;--s1:#3987e5;"
       "--s2:#d95926;--s3:#199e70;}}"
       ".t{font:13px system-ui,sans-serif;fill:var(--ink2)}"
       ".ti{font:600 15px system-ui,sans-serif;fill:var(--ink)}"
       ".sub{font:12px system-ui,sans-serif;fill:var(--mut)}"
       ".lb{font:600 12px system-ui,sans-serif}")


def mean(rho, bounded, policy, path):
    """Mean over seeds of one metric, for one cell of the 2x2."""
    vals = []
    for r in RES:
        if (r["rho"] != rho or bool(r.get("bounded", False)) != bounded
                or bool(r.get("policy", False)) != policy):
            continue
        v = r
        for k in path:
            if k not in v:
                v = None
                break
            v = v[k]
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


class Plot:
    """A single line chart.  x positions are supplied already spaced."""

    def __init__(self, title, sub, xlab, xticks, ymax, ymin=0.0, yfmt="{:.0f}"):
        self.p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"'
                  f' width="{W}" height="{H}" role="img" aria-label="{title}">',
                  f"<style>{CSS}</style>",
                  f'<rect width="{W}" height="{H}" fill="var(--sf)"/>',
                  f'<text x="26" y="24" class="ti">{title}</text>',
                  f'<text x="26" y="42" class="sub">{sub}</text>']
        self.ymin, self.ymax, self.n = ymin, ymax, len(xticks)
        for i in range(5):
            v = ymin + (ymax - ymin) * i / 4
            y = self.y(v)
            self.p.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-RGT}" y2="{y:.1f}"'
                          f' stroke="var(--gr)" stroke-width="1"/>')
            self.p.append(f'<text x="{L-10}" y="{y+4:.1f}" class="t"'
                          f' text-anchor="end">{yfmt.format(v)}</text>')
        base = self.y(ymin)
        self.p.append(f'<line x1="{L}" y1="{base:.1f}" x2="{W-RGT}" y2="{base:.1f}"'
                      f' stroke="var(--ax)" stroke-width="1"/>')
        for i, t in enumerate(xticks):
            self.p.append(f'<text x="{self.x(i):.1f}" y="{H-30}" class="t"'
                          f' text-anchor="middle">{t}</text>')
        self.p.append(f'<text x="26" y="{H-8}" class="sub">{xlab}</text>')

    def x(self, i):
        return L + (W - RGT - L) * i / max(self.n - 1, 1)

    def y(self, v):
        f = (v - self.ymin) / (self.ymax - self.ymin)
        return H - BOT - f * (H - BOT - TOP)

    def rule(self, v, label):
        y = self.y(v)
        self.p.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-RGT}" y2="{y:.1f}"'
                      f' stroke="var(--ink2)" stroke-width="1"'
                      f' stroke-dasharray="4 3"/>')
        self.p.append(f'<text x="{W-RGT+8}" y="{y+4:.1f}" class="sub">{label}</text>')

    def line(self, ys, colour, dashed=False):
        pts = [(self.x(i), self.y(v)) for i, v in enumerate(ys) if v == v]
        d = " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.p.append(f'<path d="M{d}" fill="none" stroke="var(--{colour})"'
                      f' stroke-width="2" stroke-linejoin="round"{dash}/>')
        for x, y in pts:
            self.p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6"'
                          f' fill="var(--{colour})" stroke="var(--sf)"'
                          f' stroke-width="1.5"/>')

    def legend(self, entries):
        """entries: list of (label, colour, dashed)."""
        for n, (lab, col, dash) in enumerate(entries):
            y = TOP + 10 + n * 22
            da = ' stroke-dasharray="5 4"' if dash else ""
            self.p.append(f'<line x1="{W-RGT+8}" y1="{y}" x2="{W-RGT+34}"'
                          f' y2="{y}" stroke="var(--{col})" stroke-width="2"{da}/>')
            self.p.append(f'<text x="{W-RGT+40}" y="{y+4}" class="lb"'
                          f' fill="var(--{col})">{lab}</text>')

    def save(self, name):
        open(OUT + name, "w").write("\n".join(self.p) + "\n</svg>")
        print("wrote", OUT + name)


def fig_workload():
    """AIM's own metric across the budget sweep, all four arms."""
    g = Plot("3-way workload error, all four arms",
             "3-way workload error, AIM's native metric. Mean over 5 seeds. "
             "Lower is better.",
             "zCDP budget rho (geometric, x4 per step)",
             [str(r) for r in RHOS], 0.80, 0.0, "{:.2f}")
    for bounded, pol, lab, col, dash in ARMS:
        g.line([mean(r, bounded, pol, ["wl_mean"]) for r in RHOS], col, dash)
    g.legend([(l, c, d) for _, _, l, c, d in ARMS])
    g.save("workload-error.svg")


def fig_range(ax, rho, name):
    """Absolute range error against interval width, all four arms."""
    attr, theta = ORD[ax]
    series = {(b, p): [mean(rho, b, p, ["range", ax, w]) for w in BANDS]
              for b, p, _, _, _ in ARMS}
    top = max(v for s in series.values() for v in s if v == v) * 1.12
    g = Plot(f"Range error on {attr}, by interval width",
             f"rho = {rho}, mean over 5 seeds, in records. Lower is better. "
             f"Policy threshold theta = {theta}.",
             "interval width (number of adjacent values summed)",
             BANDS, top)
    for bounded, pol, lab, col, dash in ARMS:
        g.line(series[(bounded, pol)], col, dash)
    g.legend([(l, c, d) for _, _, l, c, d in ARMS])
    g.save(name)


def fig_ratio_2x2(rho):
    """policy / stock against width, one panel per privacy definition."""
    pw, ph, gap, top, left = 250, 210, 60, 92, 62
    W2, H2 = left + 2 * pw + gap + 120, top + ph + 66
    ymax = 2.2

    def px(i, j):
        return left + i * (pw + gap) + pw * j / (len(BANDS) - 1)

    def py(v):
        return top + ph - (min(v, ymax) / ymax) * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W2} {H2}"'
         f' width="{W2}" height="{H2}" role="img"'
         f' aria-label="Range error ratio, bounded vs unbounded">',
         f"<style>{CSS}</style>",
         f'<rect width="{W2}" height="{H2}" fill="var(--sf)"/>',
         f'<text x="26" y="28" class="ti">Range error ratio, policy / stock,'
         f' within each privacy definition</text>',
         f'<text x="26" y="48" class="sub">Range error ratio, policy / stock,'
         f' within each privacy definition. rho = {rho}. Below the dashed line'
         f' the transform wins.</text>']

    for i, (bounded, rel) in enumerate([(True, "BOUNDED"), (False, "UNBOUNDED")]):
        for gl in range(5):
            v = ymax * gl / 4
            p.append(f'<line x1="{px(i,0):.1f}" y1="{py(v):.1f}"'
                     f' x2="{px(i,4):.1f}" y2="{py(v):.1f}"'
                     f' stroke="var(--gr)" stroke-width="1"/>')
            if i == 0:
                p.append(f'<text x="{left-10}" y="{py(v)+4:.1f}" class="t"'
                         f' text-anchor="end">{v:.1f}</text>')
        p.append(f'<line x1="{px(i,0):.1f}" y1="{py(1.0):.1f}"'
                 f' x2="{px(i,4):.1f}" y2="{py(1.0):.1f}" stroke="var(--ink2)"'
                 f' stroke-width="1" stroke-dasharray="4 3"/>')
        p.append(f'<text x="{(px(i,0)+px(i,4))/2:.1f}" y="{top-14}" class="ti"'
                 f' text-anchor="middle">{rel}</text>')
        for j, b in enumerate(BANDS):
            dy = 20 if j % 2 == 0 else 36
            p.append(f'<text x="{px(i,j):.1f}" y="{top+ph+dy}" class="t"'
                     f' text-anchor="middle">{b}</text>')
        for ax, colour in zip(["0", "1", "2"], ["s1", "s2", "s3"]):
            pts = []
            for j, b in enumerate(BANDS):
                s = mean(rho, bounded, False, ["range", ax, b])
                q = mean(rho, bounded, True, ["range", ax, b])
                if s == s and q == q:
                    pts.append((px(i, j), py(q / s)))
            d = " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
            p.append(f'<path d="M{d}" fill="none" stroke="var(--{colour})"'
                     f' stroke-width="2" stroke-linejoin="round"/>')
            for x, y in pts:
                p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4"'
                         f' fill="var(--{colour})" stroke="var(--sf)"'
                         f' stroke-width="1.5"/>')

    for n, (ax, colour) in enumerate(zip(["0", "1", "2"], ["s1", "s2", "s3"])):
        attr, theta = ORD[ax]
        p.append(f'<text x="{W2-112}" y="{top+18+n*20}" class="lb"'
                 f' fill="var(--{colour})">{attr.split(".")[0]} (t={theta})</text>')
    p.append(f'<text x="26" y="{H2-10}" class="sub">interval width '
             f'(number of adjacent values summed)</text>')
    open(OUT + "range-ratio-2x2.svg", "w").write("\n".join(p) + "\n</svg>")
    print("wrote", OUT + "range-ratio-2x2.svg")


def fig_heatmap():
    """The whole grid at once: ratio per attribute x relation x budget x width.

    Colour carries the pattern -- blue where the transform wins, orange where it
    loses, intensity by how far from parity.  The number stays in the cell so
    the figure is still readable as data.
    """
    cw, ch = 56, 32                         # one cell
    pw, phh = cw * len(BANDS), ch * len(RHOS)
    left, top, colgap, rowgap = 84, 74, 46, 62
    W2 = left + 2 * pw + colgap + 96
    H2 = top + 3 * (phh + rowgap) + 10

    def colour(r):
        """(css var, opacity) -- blue below parity, orange above."""
        t = np.log2(r)
        o = min(abs(t) / 1.0, 1.0) * 0.78 + 0.06
        return ("s1", o) if t < 0 else ("s2", o)

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W2} {H2}"'
         f' width="{W2}" height="{H2}" role="img"'
         f' aria-label="Range error ratio grid">',
         f"<style>{CSS}</style>",
         f'<rect width="{W2}" height="{H2}" fill="var(--sf)"/>',
         f'<text x="26" y="28" class="ti">Range error ratio, policy / stock —'
         f' every budget and width band</text>',
         f'<text x="26" y="48" class="sub">Blue: the transform has lower error'
         f' than stock under the same privacy definition. Orange: higher.'
         f' Deeper colour is further from parity.</text>']

    for row, (ax, (attr, theta)) in enumerate(ORD.items()):
        y0 = top + row * (phh + rowgap)
        for col, (bounded, rel) in enumerate([(True, "BOUNDED"),
                                              (False, "UNBOUNDED")]):
            x0 = left + col * (pw + colgap)
            p.append(f'<text x="{x0}" y="{y0-10}" class="lb"'
                     f' fill="var(--ink)">{attr} (t={theta}) — {rel}</text>')
            for i, rho in enumerate(RHOS):
                if col == 0:
                    p.append(f'<text x="{left-10}" y="{y0+i*ch+ch/2+4:.0f}"'
                             f' class="t" text-anchor="end">{rho}</text>')
                for j, b in enumerate(BANDS):
                    s = mean(rho, bounded, False, ["range", ax, b])
                    q = mean(rho, bounded, True, ["range", ax, b])
                    x, y = x0 + j * cw, y0 + i * ch
                    if s != s or q != q:
                        p.append(f'<rect x="{x}" y="{y}" width="{cw}"'
                                 f' height="{ch}" fill="none"'
                                 f' stroke="var(--gr)" stroke-width="1"/>')
                        continue
                    c, o = colour(q / s)
                    p.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}"'
                             f' fill="var(--{c})" fill-opacity="{o:.2f}"'
                             f' stroke="var(--sf)" stroke-width="1.5"/>')
                    p.append(f'<text x="{x+cw/2:.0f}" y="{y+ch/2+4:.0f}"'
                             f' class="t" fill="var(--ink)"'
                             f' text-anchor="middle">{q/s:.2f}</text>')
            for j, b in enumerate(BANDS):
                p.append(f'<text x="{x0+j*cw+cw/2:.0f}" y="{y0+phh+16}"'
                         f' class="t" text-anchor="middle">{b}</text>')

    lx = left + 2 * pw + colgap + 16
    p.append(f'<text x="{lx}" y="{top+2}" class="lb" fill="var(--ink)">ratio</text>')
    for n, v in enumerate([0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.1]):
        c, o = colour(v)
        yy = top + 14 + n * 22
        p.append(f'<rect x="{lx}" y="{yy}" width="22" height="16"'
                 f' fill="var(--{c})" fill-opacity="{o:.2f}"'
                 f' stroke="var(--gr)" stroke-width="1"/>')
        p.append(f'<text x="{lx+28}" y="{yy+12}" class="t">{v:.1f}</text>')
    p.append(f'<text x="26" y="{H2-8}" class="sub">rows: zCDP budget rho'
             f'  |  columns: interval width band</text>')
    open(OUT + "range-ratio-grid.svg", "w").write("\n".join(p) + "\n</svg>")
    print("wrote", OUT + "range-ratio-grid.svg")


if __name__ == "__main__":
    fig_workload()
    fig_range("0", 0.16, "range-age.svg")
    fig_range("1", 0.16, "range-hours.svg")
    fig_ratio_2x2(0.16)
    fig_heatmap()
