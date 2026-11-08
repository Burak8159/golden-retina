"""Render aggregate Zhejiang transfer metrics as an SVG figure."""
from __future__ import annotations
import json
from pathlib import Path

root = Path('results/technical_dev_20261107')
sampling = json.loads((root/'zhejiang_patient_sampling_sensitivity.json').read_text())
quality = json.loads((root/'zhejiang_quality_stratified_analysis.json').read_text())
W,H=1320,620
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
'<rect width="100%" height="100%" fill="#fff"/>',
'<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172033}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#4d5a70}.axis{font-size:13px}.legend{font-size:13px}.note{font-size:12px;fill:#39465b}.panel{font-size:17px;font-weight:700}</style>',
'<text x="42" y="38" class="title">Frozen DR-risk score on Zhejiang UWF: aggregation and quality sensitivity</text>',
'<text x="42" y="62" class="sub">One-shot retrospective transfer analysis · patient grouping retained · no threshold/model tuning</text>']
cats=['AMD','DR','PM','RD','RVO','Uveitis']
colors={'patient_mean':'#2563eb','patient_max':'#dc2626','first_eye':'#059669'}
labels={'patient_mean':'Mean score per patient','patient_max':'Maximum score per patient','first_eye':'One deterministic eye'}
x0,y0,pw,ph=82,150,720,350
svg.append('<text x="82" y="112" class="panel">A. Disease-vs-healthy AUROC depends on patient aggregation</text>')
def ycoord(v, top=y0, height=ph): return top+height-(v-0.25)/0.75*height
for tick in [0.25,0.4,0.55,0.7,0.85,1.0]:
    y=ycoord(tick); svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" stroke="#e5e9f0"/>'); svg.append(f'<text x="{x0-12}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>')
svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}" stroke="#667085"/><line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="#667085"/>')
xs=[x0+42+i*126 for i in range(len(cats))]
for x,c in zip(xs,cats): svg.append(f'<text x="{x}" y="{y0+ph+24}" text-anchor="middle" class="axis">{c}</text>')
for method,color in colors.items():
    pts=[]
    for x,c in zip(xs,cats):
        m=sampling['conditions'][c]['methods'][method]; v=m['auroc']; lo,hi=m['patient_bootstrap_95_percentile']['auroc_ci95']; y=ycoord(v); yl=ycoord(lo); yh=ycoord(hi)
        svg.append(f'<line x1="{x}" y1="{yh:.1f}" x2="{x}" y2="{yl:.1f}" stroke="{color}" stroke-width="2"/><line x1="{x-5}" y1="{yh:.1f}" x2="{x+5}" y2="{yh:.1f}" stroke="{color}"/><line x1="{x-5}" y1="{yl:.1f}" x2="{x+5}" y2="{yl:.1f}" stroke="{color}"/>')
        svg.append(f'<circle cx="{x}" cy="{y:.1f}" r="5" fill="{color}"/>'); pts.append(f'{x},{y:.1f}')
    svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
svg.append('<text x="110" y="535" class="legend" fill="#2563eb">● Mean per patient</text><text x="300" y="535" class="legend" fill="#dc2626">● Max per patient</text><text x="490" y="535" class="legend" fill="#059669">● One deterministic eye</text>')
rx,ry,rw,rh=875,150,380,350
svg.append('<text x="875" y="112" class="panel">B. Ranking within good-quality images</text>')
for tick in [0.25,0.4,0.55,0.7,0.85,1.0]:
    y=ry+rh-(tick-0.25)/0.75*rh; svg.append(f'<line x1="{rx}" y1="{y:.1f}" x2="{rx+rw}" y2="{y:.1f}" stroke="#e5e9f0"/>'); svg.append(f'<text x="{rx-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>')
svg.append(f'<line x1="{rx}" y1="{ry}" x2="{rx}" y2="{ry+rh}" stroke="#667085"/><line x1="{rx}" y1="{ry+rh}" x2="{rx+rw}" y2="{ry+rh}" stroke="#667085"/>')
qxs=[rx+34+i*60 for i in range(len(cats))]
for x,c in zip(qxs,cats): svg.append(f'<text x="{x}" y="{ry+rh+24}" text-anchor="middle" class="axis">{c}</text>')
for key,color in [('all','#64748b'),('good_quality','#7c3aed')]:
    pts=[]
    for x,c in zip(qxs,cats):
        m=quality['conditions'][c]['strata'][key]; v=m['auroc']; lo,hi=m['ci95']; y=ry+rh-(v-.25)/.75*rh; yl=ry+rh-(lo-.25)/.75*rh; yh=ry+rh-(hi-.25)/.75*rh
        svg.append(f'<line x1="{x}" y1="{yh:.1f}" x2="{x}" y2="{yl:.1f}" stroke="{color}" stroke-width="2"/><circle cx="{x}" cy="{y:.1f}" r="4.5" fill="{color}"/>'); pts.append(f'{x},{y:.1f}')
    svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.2"/>')
svg.append('<text x="900" y="535" class="legend" fill="#64748b">● All quality</text><text x="1045" y="535" class="legend" fill="#7c3aed">● Good overall quality</text>')
svg.append('<text x="42" y="578" class="note">DR is the only trained disease target; other categories are exploratory transfer scores. Case counts are enriched, not prevalence estimates.</text>')
svg.append('<text x="42" y="599" class="note">Quality-stratified estimates are image-level AUROC with patient-cluster bootstrap; poor-quality controls are sparse (n=3).</text>')
svg.append('</svg>')
out=root/'zhejiang_transfer_sampling_quality.svg'; out.write_text('\n'.join(svg)); print(out)
