"""Regenerate the report figures from sweep.json into docs/figures/.

Two metrics are plotted, and only two: AIM's own workload error, and range
query error stratified by interval width.  Everything the report claims has to
be visible in one of these.

    python experiment/evaluation/figures.py     # from the project root
"""
import json

import numpy as np

RES = json.load(open("experiment/results/sweep.json"))
OUT = "docs/figures/"

RHOS = sorted({r["rho"] for r in RES})
BANDS = ["1-2", "3-5", "6-10", "11-20", ">20"]
ORD = {"0": ("age", 7), "1": ("hours.per.week", 8), "2": ("education.num", 2)}

W, H = 680, 380
L, RGT, TOP, BOT = 66, 118, 66, 52          # margins: plot box is what's left

CSS = (":root{--sf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;"
       "--gr:#e1e0d9;--ax:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;}"
       "@media (prefers-color-scheme:dark){:root{--sf:#1a1a19;--ink:#ffffff;"
       "--ink2:#c3c2b7;--mut:#898781;--gr:#2c2c2a;--ax:#383835;--s1:#3987e5;"
       "--s2:#d95926;--s3:#199e70;}}"
       ".t{font:13px system-ui,sans-serif;fill:var(--ink2)}"
       ".ti{font:600 15px system-ui,sans-serif;fill:var(--ink)}"
       ".sub{font:12px system-ui,sans-serif;fill:var(--mut)}"
       ".lb{font:600 12px system-ui,sans-serif}")


def rows(rho, pol):
    return [r for r in RES
            if r["rho"] == rho and bool(r.get("policy", False)) == pol]


def mean(rho, pol, path):
    """Mean over seeds of one metric.  Missing bands are skipped."""
    vals = []
    for r in rows(rho, pol):
        v = r
        for k in path:
            if k not in v:
                v = None
                break
            v = v[k]
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


# ------------------------------------------------------------------ plotting
class Plot:
    """A single line chart.  x positions are supplied already spaced."""

    def __init__(self, title, sub, xlab, xticks, ymax, ymin=0.0, yfmt="{:.0f}"):
        self.p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"'
                  f' width="{W}" height="{H}" role="img" aria-label="{title}">',
                  f"<style>{CSS}</style>",
                  f'<rect width="{W}" height="{H}" fill="var(--sf)"/>',
                  f'<text x="26" y="24" class="ti">{title}</text>',
                  f'<text x="26" y="42" class="sub">{sub}</text>']
        self.ymin, self.ymax = ymin, ymax
        self.n = len(xticks)

        for i in range(5):                          # horizontal gridlines
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
        """A horizontal reference line, e.g. ratio = 1."""
        y = self.y(v)
        self.p.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-RGT}" y2="{y:.1f}"'
                      f' stroke="var(--ink2)" stroke-width="1"'
                      f' stroke-dasharray="4 3"/>')
        self.p.append(f'<text x="{W-RGT+8}" y="{y+4:.1f}" class="sub">{label}</text>')

    def mark(self, i, label):
        """A vertical annotation at tick i."""
        self.p.append(f'<line x1="{self.x(i):.1f}" y1="{TOP-8}" x2="{self.x(i):.1f}"'
                      f' y2="{self.y(self.ymin):.1f}" stroke="var(--ax)"'
                      f' stroke-width="1" stroke-dasharray="2 4"/>')
        self.p.append(f'<text x="{self.x(i):.1f}" y="{TOP-14}" class="sub"'
                      f' text-anchor="middle">{label}</text>')

    def line(self, ys, colour, label, dy=4):
        pts = [(self.x(i), self.y(v)) for i, v in enumerate(ys) if v == v]
        d = " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        self.p.append(f'<path d="M{d}" fill="none" stroke="var(--{colour})"'
                      f' stroke-width="2" stroke-linejoin="round"'
                      f' stroke-linecap="round"/>')
        for x, y in pts:
            self.p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5"'
                          f' fill="var(--{colour})" stroke="var(--sf)"'
                          f' stroke-width="2"/>')
        lx, ly = pts[-1]
        self.p.append(f'<text x="{lx+11:.1f}" y="{ly+dy:.1f}" class="lb"'
                      f' fill="var(--{colour})">{label}</text>')

    def save(self, name):
        open(OUT + name, "w").write("\n".join(self.p) + "\n</svg>")
        print("wrote", OUT + name)


# --------------------------------------------------------------- the figures
def fig_workload():
    """AIM's own metric across the budget sweep."""
    s = [mean(r, False, ["wl_mean"]) for r in RHOS]
    p = [mean(r, True, ["wl_mean"]) for r in RHOS]
    g = Plot("Cell-level accuracy: the policy is worse at every budget",
             "3-way workload error, AIM's native metric. Mean over 5 seeds. "
             "Lower is better.",
             "zCDP budget rho (geometric, x4 per step)",
             [str(r) for r in RHOS], 0.80, 0.0, "{:.2f}")
    g.line(s, "s1", "stock", dy=12)
    g.line(p, "s2", "policy", dy=-4)
    g.save("workload-error.svg")


def fig_range(ax, rho, name):
    """Absolute range error against interval width, both arms."""
    attr, theta = ORD[ax]
    s = [mean(rho, False, ["range", ax, b]) for b in BANDS]
    p = [mean(rho, True, ["range", ax, b]) for b in BANDS]
    top = max(max(s), max(p)) * 1.15
    g = Plot(f"Range-query error on {attr}: the lines cross",
             f"rho = {rho}, mean over 5 seeds. Lower is better. "
             f"Policy threshold theta = {theta}.",
             "interval width (number of adjacent values summed)",
             BANDS, top)
    g.line(s, "s1", "stock")
    g.line(p, "s2", "policy")
    g.save(name)


def fig_ratio(rho):
    """policy / stock against width, all three ordinal attributes."""
    g = Plot("The policy wins only above a width threshold",
             f"Range error ratio, policy / stock, at rho = {rho}. "
             "Below 1.0 the policy wins.",
             "interval width (number of adjacent values summed)",
             BANDS, 2.2, 0.0, "{:.1f}")
    g.rule(1.0, "parity")
    for ax, colour, dy in zip(["0", "1", "2"], ["s1", "s2", "s3"], [4, 4, -8]):
        attr, theta = ORD[ax]
        r = [mean(rho, True, ["range", ax, b]) / mean(rho, False, ["range", ax, b])
             for b in BANDS]
        g.line(r, colour, f"{attr.split('.')[0]} (t={theta})", dy=dy)
    g.save("range-ratio.svg")


def fig_ratio_panels():
    """One panel per budget: ratio against width, all three ordinal attributes.

    Small multiples rather than five separate files -- the point is to compare
    across budgets, which needs them side by side and on a shared y-scale.
    """
    pw, ph = 250, 210                       # one panel's plot box
    gap, top, left = 34, 76, 62
    W2 = left + 5 * pw + 4 * gap + 96
    H2 = top + ph + 62
    ymax = 2.2

    def px(i, j):                           # panel i, tick j
        x0 = left + i * (pw + gap)
        return x0 + pw * j / (len(BANDS) - 1)

    def py(v):
        return top + ph - (v / ymax) * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W2} {H2}"'
         f' width="{W2}" height="{H2}" role="img"'
         f' aria-label="Range error ratio by width, at each budget">',
         f"<style>{CSS}</style>",
         f'<rect width="{W2}" height="{H2}" fill="var(--sf)"/>',
         f'<text x="26" y="28" class="ti">The width effect holds at every '
         f'budget — but not by the same amount</text>',
         f'<text x="26" y="48" class="sub">Range error ratio, policy / stock. '
         f'Below the dashed line the policy wins. Shared scale.</text>']

    for i, rho in enumerate(RHOS):
        for g in range(5):                  # gridlines
            v = ymax * g / 4
            p.append(f'<line x1="{px(i,0):.1f}" y1="{py(v):.1f}"'
                     f' x2="{px(i,4):.1f}" y2="{py(v):.1f}"'
                     f' stroke="var(--gr)" stroke-width="1"/>')
            if i == 0:
                p.append(f'<text x="{left-10}" y="{py(v)+4:.1f}" class="t"'
                         f' text-anchor="end">{v:.1f}</text>')
        p.append(f'<line x1="{px(i,0):.1f}" y1="{py(1.0):.1f}"'
                 f' x2="{px(i,4):.1f}" y2="{py(1.0):.1f}" stroke="var(--ink2)"'
                 f' stroke-width="1" stroke-dasharray="4 3"/>')
        p.append(f'<text x="{(px(i,0)+px(i,4))/2:.1f}" y="{top-12}" class="lb"'
                 f' fill="var(--ink)" text-anchor="middle">rho = {rho}</text>')
        for j, b in enumerate(BANDS):       # x labels, alternating height
            dy = 20 if j % 2 == 0 else 36
            p.append(f'<text x="{px(i,j):.1f}" y="{top+ph+dy}" class="t"'
                     f' text-anchor="middle">{b}</text>')

        for ax, colour in zip(["0", "1", "2"], ["s1", "s2", "s3"]):
            pts = []
            for j, b in enumerate(BANDS):
                s = mean(rho, False, ["range", ax, b])
                q = mean(rho, True, ["range", ax, b])
                if s == s and q == q:
                    pts.append((px(i, j), py(min(q / s, ymax))))
            d = " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
            p.append(f'<path d="M{d}" fill="none" stroke="var(--{colour})"'
                     f' stroke-width="2" stroke-linejoin="round"/>')
            for x, y in pts:
                p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2"'
                         f' fill="var(--{colour})" stroke="var(--sf)"'
                         f' stroke-width="1.5"/>')

    for n, (ax, colour) in enumerate(zip(["0", "1", "2"], ["s1", "s2", "s3"])):
        attr, theta = ORD[ax]
        p.append(f'<text x="{W2-92}" y="{top+18+n*20}" class="lb"'
                 f' fill="var(--{colour})">{attr.split(".")[0]} (t={theta})</text>')
    p.append(f'<text x="26" y="{H2-10}" class="sub">interval width '
             f'(number of adjacent values summed)</text>')
    open(OUT + "range-ratio-panels.svg", "w").write("\n".join(p) + "\n</svg>")
    print("wrote", OUT + "range-ratio-panels.svg")


if __name__ == "__main__":
    fig_workload()
    fig_range("0", 0.16, "range-age.svg")
    fig_range("1", 0.16, "range-hours.svg")
    fig_ratio(0.16)
    fig_ratio_panels()
