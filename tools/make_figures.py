"""Generate descriptive figures directly from the release CSVs. No model experiments."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(name):
    with (ROOT/'data'/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
    rows=read('dictionary.csv');examples=read('examples.csv');layers=[12,20,32,44]
    totals=[sum(int(r['layer'])==l for r in rows) for l in layers]
    changed=[sum(r['entry_id'].startswith(f'L{l}_') and r['excerpt_differs']=='true' for r in examples) for l in layers]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5.5),facecolor='#f7f8fc');ax.set_facecolor('#f7f8fc')
    x=range(4)
    ax.bar(x,totals,color='#d4dcec',width=.58,label='Sampled entries')
    ax.bar(x,changed,color='#3457d5',width=.58,label='Different paired excerpts')
    for i,(n,c) in enumerate(zip(totals,changed)):
        ax.text(i,n+7,str(n),ha='center',fontweight='bold',color='#27334b')
        ax.text(i,c/2,f'{c}\n{c/n:.1%}',ha='center',va='center',color='white',fontweight='bold')
    ax.set_xticks(list(x),[f'Layer {l}' for l in layers]);ax.set_ylabel('Channel entries');ax.set_ylim(0,300)
    fig.suptitle('Qwen3.8-27B-Neuron Dictionary',x=.075,y=.96,ha='left',fontweight='bold',fontsize=21,color='#17213b')
    ax.set_title('704 sampled channels · 16 fixed probes · 187 different excerpt pairs',loc='left',pad=19,color='#4c5870',fontsize=12)
    ax.legend(frameon=False,loc='upper right')
    fig.text(.075,.035,'Exact excerpt differences, not changed factual answers. Selected probes; bounded generations.',fontsize=10,color='#4c5870')
    fig.subplots_adjust(left=.09,right=.97,top=.79,bottom=.16)
    (ROOT/'assets').mkdir(exist_ok=True)
    fig.savefig(ROOT/'assets/dataset-overview.png',dpi=180,facecolor=fig.get_facecolor())
    fig.savefig(ROOT/'assets/dataset-overview.svg',facecolor=fig.get_facecolor())
    plt.close(fig)
if __name__=='__main__':main()
