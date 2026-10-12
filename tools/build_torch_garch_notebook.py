import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'archive/01_data_clean.py').read_text(encoding='utf-8').split('# %% [markdown]\n# ## Phần tiếp theo')[0]
base=base.replace('Đây là tầng dự báo của Bước 1. Robust minimum variance và backtest danh mục\n# sẽ được bổ sung sau khi định nghĩa cách liên kết bất định với tối ưu.',
                  'Bản mở rộng Bước 1: Ridge, MLP PyTorch, GARCH arch và robust MV.')
extension=(ROOT/'src/torch_garch_extension.py').read_text(encoding='utf-8')
helper=(ROOT/'src/portfolio_baseline.py').read_text(encoding='utf-8')
extension=extension.replace('from portfolio_baseline import minimum_variance, performance',helper)
source=base+extension
ast.parse(source)
(ROOT/'src/03_torch_garch_baseline.py').write_text(source,encoding='utf-8')
cells=[]
for chunk in source.split('# %%')[1:]:
    markdown=chunk.startswith(' [markdown]')
    lines=chunk.splitlines()[1:]
    if markdown:
        text='\n'.join(line[2:] if line.startswith('# ') else line[1:] if line.startswith('#') else line for line in lines)
        cell={'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
    else:
        text='\n'.join(lines)+'\n'
        ast.parse(text)
        cell={'cell_type':'code','execution_count':None,'outputs':[],'metadata':{},'source':text.splitlines(keepends=True)}
    cell['id']=f'torch-garch-{len(cells):02d}'
    cells.append(cell)
nb={'cells':cells,'nbformat':4,'nbformat_minor':5,
    'metadata':{'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'},'language_info':{'name':'python'}}}
(ROOT/'03_torch_garch_baseline.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print('Built standalone script and notebook')
