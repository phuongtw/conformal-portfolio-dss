"""Focused checks for forecast timing, units and portfolio constraints."""
import ast
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from arch import arch_model
from portfolio_baseline import minimum_variance, performance

tree=ast.parse((ROOT / 'src/torch_garch_extension.py').read_text(encoding='utf-8'))
names={'fit_garch_and_forecast','extended_backtest'}
definitions=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names], type_ignores=[])
scope={'np':np,'pd':pd,'arch_model':arch_model,'minimum_variance':minimum_variance}
exec(compile(definitions,'extension_functions','exec'),scope)
prices=pd.read_csv(ROOT / 'data/etf_adjusted_close_2010_2025.csv',index_col=0,parse_dates=True)
tickers=['SPY','QQQ','TLT','GLD','EEM']
returns=prices[tickers].pct_change(fill_method=None).iloc[1:]
train=returns.loc['2010':'2018']
future=returns.loc['2019'].iloc[:12]
fit=scope['fit_garch_and_forecast']
sigma,params,states=fit(train,future)
changed=future.copy()
changed.iloc[5:]=.25
other,_,_=fit(train,changed)
pd.testing.assert_frame_equal(sigma.iloc[:6],other.iloc[:6])
assert not np.allclose(sigma.iloc[6:],other.iloc[6:])
for asset in tickers:
    p=states[asset]
    assert np.isclose(sigma[asset].iloc[0]**2,p['initial_h']/10000)
    h_next=p['omega']+p['a']*(future[asset].iloc[0]*100-p['mu'])**2+p['b']*p['initial_h']
    assert np.isclose(sigma[asset].iloc[1]**2,h_next/10000)
scope['TICKERS']=tickers
q=pd.Series(.015,index=tickers)
scores=pd.DataFrame(np.tile(np.linspace(.001,.02,12)[:,None],(1,5)),columns=tickers,index=future.index)
width=sigma*2
test=scope['extended_backtest']
records,weights=test(returns,future.index,sigma,width,q,q,scores,scores/sigma)
changed_returns=returns.copy()
changed_returns.loc[changed_returns.index>future.index[-1]]=.3
other_records,other_weights=test(changed_returns,future.index,sigma,width,q,q,scores,scores/sigma)
pd.testing.assert_frame_equal(records,other_records)
pd.testing.assert_frame_equal(weights,other_weights)
assert np.allclose(weights[tickers].sum(axis=1),1)
assert (weights[tickers]>=0).all().all()
assert np.allclose(records.Net,(1-records.Cost)*(1+records.Gross)-1)
assert (records.Cost>=0).all()
print('Passed GARCH timing, percent/decimal conversion, future-data invariance, portfolio constraints and cost formula.')
