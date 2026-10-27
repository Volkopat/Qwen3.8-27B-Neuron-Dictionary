"""Browse archived entries without loading model weights."""
import argparse,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):
    with (ROOT/'data'/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--layer',type=int);p.add_argument('--entry');p.add_argument('--limit',type=int,default=10);a=p.parse_args()
    rows=read('dictionary.csv');examples={r['entry_id']:r for r in read('examples.csv')}
    if a.entry:rows=[r for r in rows if r['entry_id']==a.entry]
    if a.layer is not None:rows=[r for r in rows if int(r['layer'])==a.layer]
    rows=sorted(rows,key=lambda r:-float(r['max_abs_first_token_delta_nats']))
    if not rows:p.error('no matching entries')
    for r in rows[:max(1,a.limit)]:print(json.dumps({'dictionary':r,'example':examples[r['entry_id']]},ensure_ascii=False))
if __name__=='__main__':main()
