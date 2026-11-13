"""DENG611 Assessment 3: reproducible biodigester anomaly screening.

Run: python DENG611_Assessment3_Simulation_REVISED.py
Install: pip install pandas openpyxl numpy scikit-learn matplotlib
Place this script in the same folder as Farm-scale Biodigester 2024_Performance(1).xlsx
(or pass the workbook path as the first command-line argument).

Creates DENG611_figures/figure1_timeseries.png ... figure4_robustness.png,
cleaned_biogas_data.csv, test_predictions.csv, sensitivity.csv, robustness.csv.
Unsupervised alerts are not confirmed failures. Sensor perturbations are simulations.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Save images even when no display is available.
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'DENG611_figures'
OUT.mkdir(exist_ok=True)
DEFAULT = ROOT / 'Farm-scale Biodigester 2024_Performance(1).xlsx'
WORKBOOK = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else DEFAULT
if not WORKBOOK.is_file():
    matches = sorted(ROOT.glob('Farm-scale Biodigester 2024_Performance*.xlsx'))
    if not matches:
        raise FileNotFoundError(f'Workbook not found: {WORKBOOK}. Put it next to this script or pass its path.')
    WORKBOOK = matches[0]

frames = []
for sheet, temp_col in [('RI-FLEX','DigesterTemp (C)09:25am'),('R2-FLEX','DigesterTemp (C)time 09:25am')]:
    source = pd.read_excel(WORKBOOK, sheet_name=sheet)
    frame = pd.DataFrame({
        'digester': sheet,
        'date': pd.to_datetime(source['Date (m/d/y)'], errors='coerce'),
        'air_temp_c': pd.to_numeric(source['Air Temp (C)time 09:15am'],errors='coerce'),
        'digester_temp_c': pd.to_numeric(source[temp_col],errors='coerce'),
        'biogas_stp_ml': pd.to_numeric(source['Daily Biogas at STP (mL)'],errors='coerce'),
        'water_added_kg': pd.to_numeric(source['Water added (kg)'],errors='coerce'),
        'manure_fed_kg': pd.to_numeric(source['Manure fed (kg)'].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0],errors='coerce'),
    })
    frames.append(frame)
raw = pd.concat(frames,ignore_index=True)
df = raw.dropna(subset=['date','biogas_stp_ml']).copy()
df = df[df.biogas_stp_ml.between(0,100000)].copy()
for col,lo,hi in [('air_temp_c',-10,60),('digester_temp_c',0,65),('water_added_kg',0,1000),('manure_fed_kg',0,1000)]:
    df.loc[~df[col].between(lo,hi),col] = np.nan
df = df.sort_values(['date','digester']).reset_index(drop=True)
df.to_csv(OUT/'cleaned_biogas_data.csv',index=False)
features = ['air_temp_c','digester_temp_c','biogas_stp_ml','water_added_kg','manure_fed_kg']
dates = np.sort(df.date.unique())
cutoff = dates[int(len(dates)*.70)]
train = df[df.date < cutoff].copy()
test = df[df.date >= cutoff].copy()

def fit_model(contamination):
    model = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('forest',IsolationForest(n_estimators=200,contamination=contamination,random_state=42))
    ])
    model.fit(train[features])
    return model

model = fit_model(.05)
for subset in (train,test):
    subset['score'] = -model.decision_function(subset[features])
    subset['alert'] = model.predict(subset[features]) == -1

sensitivity = []
for contamination in (.02,.05,.10):
    fitted = fit_model(contamination)
    flagged = fitted.predict(test[features]) == -1
    sensitivity.append({'contamination':contamination,'test_alerts':int(flagged.sum()),'test_alert_rate_pct':round(100*flagged.mean(),2)})
pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity.csv',index=False)

rng = np.random.default_rng(42)
missing = test[features].copy()
missing = missing.mask(rng.random(missing.shape)<.20)
missing_alert = model.predict(missing) == -1
offset = test[features].copy()
offset['digester_temp_c'] += 3
shift_alert = model.predict(offset) == -1
original = test.alert.to_numpy()
robustness = pd.DataFrame([
    {'scenario':'Original','alerts':int(original.sum()),'disagreements_vs_original':0},
    {'scenario':'20% missing','alerts':int(missing_alert.sum()),'disagreements_vs_original':int(np.sum(missing_alert != original))},
    {'scenario':'Temp +3 C','alerts':int(shift_alert.sum()),'disagreements_vs_original':int(np.sum(shift_alert != original))},
])
robustness.to_csv(OUT/'robustness.csv',index=False)
test[['date','digester']+features+['score','alert']].to_csv(OUT/'test_predictions.csv',index=False)

fig,ax = plt.subplots(figsize=(11,5))
for digester,group in test.groupby('digester'):
    line, = ax.plot(group.date,group.biogas_stp_ml,'.-',alpha=.65,label=digester)
    alerts = group[group.alert]
    ax.scatter(alerts.date,alerts.biogas_stp_ml,s=85,marker='x',color=line.get_color(),linewidth=1.8,label=f'{digester} alert')
ax.set(title='Figure 1. Held-out biogas production and anomaly alerts',xlabel='Date',ylabel='Daily biogas at STP (mL)')
ax.legend(ncol=2,fontsize=9);ax.grid(alpha=.2);fig.autofmt_xdate();fig.tight_layout()
fig.savefig(OUT/'figure1_timeseries.png',dpi=180);plt.close(fig)

fig,ax = plt.subplots(figsize=(11,5))
for label,group in [('Normal',test[~test.alert]),('Alert',test[test.alert])]:
    ax.scatter(group.date,group.score,s=22 if label=='Normal' else 35,alpha=.8,label=label)
ax.axhline(0,color='black',linestyle='--',label='Alert boundary')
ax.set(title='Figure 2. Isolation Forest scores on held-out records',xlabel='Date',ylabel='Anomaly score (higher = more unusual)')
ax.legend();ax.grid(alpha=.2);fig.autofmt_xdate();fig.tight_layout()
fig.savefig(OUT/'figure2_scores.png',dpi=180);plt.close(fig)

fig,ax = plt.subplots(figsize=(8,5))
labels = [f'{r["contamination"]:.0%}' for r in sensitivity]
values = [r['test_alert_rate_pct'] for r in sensitivity]
bars = ax.bar(labels,values)
for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.3,f'{value:.2f}%',ha='center')
ax.set_ylim(0,max(values)*1.18);ax.set(title='Figure 3. Sensitivity to contamination assumption',xlabel='Training contamination',ylabel='Held-out alert rate (%)')
fig.tight_layout();fig.savefig(OUT/'figure3_sensitivity.png',dpi=180);plt.close(fig)

fig,ax = plt.subplots(figsize=(8,5))
bars = ax.bar(robustness.scenario,robustness.alerts)
for bar,value in zip(bars,robustness.alerts):ax.text(bar.get_x()+bar.get_width()/2,value+.3,str(value),ha='center')
ax.set_ylim(0,max(robustness.alerts)*1.18);ax.set(title='Figure 4. Alerts under simulated sensor-data perturbations',ylabel='Held-out alerts')
fig.tight_layout();fig.savefig(OUT/'figure4_robustness.png',dpi=180);plt.close(fig)

print(f'Workbook: {WORKBOOK.name}')
print(f'Raw rows: {len(raw)}; cleaned: {len(df)}; train: {len(train)}; test: {len(test)}')
print(f'Train alerts: {train.alert.sum()}; test alerts: {test.alert.sum()} ({test.alert.mean():.2%})')
print('Sensitivity:\n',pd.DataFrame(sensitivity).to_string(index=False))
print('Robustness:\n',robustness.to_string(index=False))
print(f'Figures and CSVs saved to: {OUT.resolve()}')
