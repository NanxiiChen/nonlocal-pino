"""Check real CLI resolution/dispatch without starting an unchanged training run."""
from pathlib import Path
import argparse,ast,json,sys
from unittest.mock import patch
from datetime import datetime,timezone
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--legacy',action='store_true');args=parser.parse_args()
from examples.confined_ch2d import train
from examples.confined_ch2d.problem import validate_config,load_reference
from nonlocal_pinn.training.runtime import with_run_seed
argv=['train.py']
if args.legacy:argv.append('--config='+str(ROOT/'examples/confined_ch2d/configs/train_base.py'))
train.flags.FLAGS(argv)
c=train._CONFIG.value
validate_config(c);ref=load_reference(c)
assert 'seed' not in c.train
if args.legacy:
    assert 'windows' not in c.train and c.physical.terminal_time==1.
    print('PASS: explicit legacy CLI config/reference, old simple physical problem preserved')
else:
    assert c.train.windows.count==30 and c.physical.terminal_time==1.5
    assert c.train.iterations==5000 and c.train.refinement.iterations==500
    assert c.log.run_name==''
    saved=json.loads(Path('/root/autodl-tmp/tf-logs/confined_ch2d/20261027-chopt-windows-01/config.json').read_text())
    saved['train'].pop('seed');saved['log']['run_name']=''
    assert saved==json.loads(c.to_json())
    runtime=with_run_seed(c,None)
    with patch('examples.confined_ch2d.window_training.train_windows',return_value='window-dispatched') as dispatch,patch.object(train,'datetime') as clock:
        clock.now.return_value=datetime(2026,10,28,12,34,56,123456,tzinfo=timezone.utc)
        assert train.train(runtime)=='window-dispatched'
        resolved=dispatch.call_args.args[0]
        assert resolved.log.run_name=='20261028-123456-123456-nonlocal-windows'
        assert resolved.train.seed==42 and runtime.log.run_name=='' and c.log.run_name==''
        runtime.log.run_name='explicit-window-run'
        assert train.train(runtime)=='window-dispatched'
        assert dispatch.call_args.args[0].log.run_name=='explicit-window-run'
    tree=ast.parse((ROOT/'examples/confined_ch2d/solver.py').read_text())
    defaults=[kw.value.value for node in ast.walk(tree) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument'
        and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='--config'
        for kw in node.keywords if kw.arg=='default']
    assert defaults==['examples.confined_ch2d.configs.train_coarsening_windows']
    print('PASS: real no-config CLI selects verified30-window/T1.5 preset; exact historical strategy parity; generated runtime name/copy, explicit name preserved; solver physical default aligned')
