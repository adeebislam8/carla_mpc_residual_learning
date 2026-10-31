#!/usr/bin/env python
"""Write a routes XML with only the given route ids (one 'RouteScenario_<id>' or '<id>' per line).

    python3 tools/b2d/make_route_subset.py --xml tools/b2d/routes/train_clips2.xml \
        --ids tools/b2d/dev_basefix_ids.txt --out tools/b2d/routes/dev_basefix.xml
"""
import argparse
import xml.etree.ElementTree as ET

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--xml', required=True)
ap.add_argument('--ids', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
want = {l.strip().replace('RouteScenario_', '').split('_')[0] for l in open(a.ids) if l.strip()}
tree = ET.parse(a.xml)
root = tree.getroot()
kept = 0
for r in list(root):
    if r.get('id') in want:
        kept += 1
    else:
        root.remove(r)
tree.write(a.out)
print(f'{kept} of {len(want)} routes -> {a.out}')
