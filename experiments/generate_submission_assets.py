"""Create vector, data-backed figures for the revised submission manuscript."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'/'pdf'/'assets'; OUT.mkdir(parents=True,exist_ok=True)

def export(fig, name):
    """Write print-quality PNGs for the locally available PDF compositor and vector PDFs for source reuse."""
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT/f'{name}.png', bbox_inches='tight', dpi=360)
    plt.close(fig)

def pipeline():
    fig,ax=plt.subplots(figsize=(12,5)); ax.set_xlim(0,12);ax.set_ylim(0,5);ax.axis('off')
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05,rounding_size=0.12',fc=color,ec='#26364A',lw=1.5))
        ax.text(x+w/2,y+h-.33,title,ha='center',va='center',fontsize=10,fontweight='bold',color='#152536')
        ax.text(x+w/2,y+h/2-.15,body,ha='center',va='center',fontsize=8.4,color='#22334A',wrap=True)
    ax.add_patch(FancyBboxPatch((.25,.35),11.5,4.3,boxstyle='round,pad=.08,rounding_size=.16',fc='none',ec='#9BA8B5',lw=1.6,ls=(0,(6,4))))
    ax.text(.45,4.35,'Frozen evaluation protocol',fontsize=13,fontweight='bold',color='#14213D')
    box(.6,2.3,2,1.25,'API-executed trajectories','Held-out normal tasks\nand six attack families','#E8F0FA')
    box(3.1,2.3,2,1.25,'Cross-session state','Decayed transition counts\nand benign frequency baseline','#DCEBFA')
    box(5.6,2.3,2,1.25,'Trajectory-only score','Transition + frequency\nsignals; no parameter rules','#E8F0FA')
    box(8.1,2.3,2,1.25,'Frozen calibration','Threshold = max of\nbenign validation maxima','#DCEBFA')
    box(10.6,2.3,.8,1.25,'Decision','Alert\n/\nWatch','#E8F0FA')
    for x in (2.6,5.1,7.6,10.1): ax.add_patch(FancyArrowPatch((x,2.92),(x+.42,2.92),arrowstyle='-|>',mutation_scale=14,lw=1.5,color='#26364A'))
    box(1.5,.78,3.0,.8,'Development evidence (R1-R3)','Used only for diagnosis and candidate selection','#F8EEDB')
    box(6.1,.78,4.6,.8,'Independent confirmation (R4-R5)','New wording, operands, API trajectories, and long normal controls','#E4F3E8')
    ax.add_patch(FancyArrowPatch((3.1,1.6),(4.2,2.25),arrowstyle='-|>',mutation_scale=14,lw=1.25,color='#8A6D3B'))
    ax.add_patch(FancyArrowPatch((8.4,1.6),(8.9,2.25),arrowstyle='-|>',mutation_scale=14,lw=1.25,color='#38704F'))
    export(fig, 'figure1_protocol')

def ecdf(values):
    x=sorted(values);return x,[(i+1)/len(x) for i in range(len(x))]
def curves():
    fig,ax=plt.subplots(figsize=(7.2,4.5)); colors={'R4 attack':'#C47D2B','R5 attack':'#8B4A28','R4 benign':'#547A9B','R5 benign':'#7899B5'}
    for run in ('R4','R5'):
        d=json.loads((ROOT/'local_results'/'canonical'/f'confirmation_{run}'/'confirmation_evaluation.json').read_text())
        for kind,label in (('attack',f'{run} attack groups'),('benign',f'{run} benign groups')):
            x,y=ecdf(list(d['scores'][kind].values()));ax.step(x,y,where='post',lw=2.4,label=label,color=colors[f'{run} {kind}'])
        ax.axvline(d['threshold'],color='#46515C',ls='--',lw=1,alpha=.55)
    ax.set_xlabel('Maximum trajectory score');ax.set_ylabel('Empirical cumulative distribution');ax.set_ylim(0,1.02);ax.grid(alpha=.25);ax.legend(frameon=True,fontsize=8,loc='lower right')
    export(fig, 'figure2_score_ecdf')

def metrics():
    fig,axs=plt.subplots(1,2,figsize=(8.2,3.5)); runs=['R4','R5']; x=range(2);w=.34
    primary=[.41875,.4375];official=[1,1];fprp=[0,.005];fpro=[.005,0]
    for ax,a,b,title,ylim in ((axs[0],primary,official,'Successful-attack recall',(0,1.05)),(axs[1],fprp,fpro,'False-positive rate',(0,.03))):
        ax.bar([i-w/2 for i in x],a,w,label='Trajectory-only candidate',color='#3B78A6');ax.bar([i+w/2 for i in x],b,w,label='Official AgentShield*',color='#D68A2F')
        ax.set_xticks(list(x),runs);ax.set_ylim(*ylim);ax.set_title(title);ax.set_ylabel('Rate');ax.grid(axis='y',alpha=.22)
    axs[0].legend(fontsize=7,frameon=True);fig.text(.5,-.04,'* Separate defended executions with honeytools/honeytokens; descriptive comparison only.',ha='center',fontsize=7)
    export(fig, 'figure3_confirmation')

def ablation():
    labels=['Trajectory-only','No cross-session','No transition/\nfrequency','No cumulative\n(exploratory)'];r4=[.2833,.2792,0,.95];r5=[.2958,.2917,0,1]
    fig,ax=plt.subplots(figsize=(7.2,4));x=range(4);w=.34;ax.bar([i-w/2 for i in x],r4,w,label='R4',color='#4E89B7');ax.bar([i+w/2 for i in x],r5,w,label='R5',color='#B56C3D');ax.set_xticks(list(x),labels);ax.set_ylim(0,1.05);ax.set_ylabel('All-attempt detection rate');ax.set_title('Mechanism analysis under validation-maximum calibration');ax.grid(axis='y',alpha=.22);ax.legend(frameon=True)
    export(fig, 'figure4_ablation')
if __name__=='__main__': pipeline();curves();metrics();ablation()
