"""Explicitly test the author's matrix path at Torch's ordinary FP32 default."""
from pathlib import Path
import importlib.util,json
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
out=ROOT/'reports/soap-implementation-audit'
spec=importlib.util.spec_from_file_location('author_reference',out/'sources/author-soap.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_default_dtype(torch.float32)
p=torch.nn.Parameter(torch.zeros((2,3),dtype=torch.float64));p.grad=torch.ones_like(p)
optimizer=module.SOAP([p],lr=.001,weight_decay=0.)
try:
 optimizer.step()
except RuntimeError as error:
 if 'dtype' not in str(error).lower():raise
 result=dict(status='observed_dtype_incompatibility',parameter_dtype=str(p.dtype),default_dtype=str(torch.get_default_dtype()),error=str(error))
else:
 raise AssertionError('Expected FP64/default-FP32 Gram allocation dtype incompatibility was not reproduced')
(out/'author-default-dtype.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
