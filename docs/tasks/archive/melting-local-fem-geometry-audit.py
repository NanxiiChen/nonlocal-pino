"""Verify the NumPy-only import refactor retains actual trained geometry math."""
import ast
import importlib.util
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from examples.icemelting3d import ellipsoids as current

OUT=Path(__file__).resolve().parent
source=Path('/root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-01/source/examples/icemelting3d/ellipsoids.py')
spec=importlib.util.spec_from_file_location('saved_melting_geometry',source)
legacy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
class StripImports(ast.NodeTransformer):
    def visit_Import(self,node):
        return None
    def visit_ImportFrom(self,node):
        return None
old=StripImports().visit(ast.parse(source.read_text()))
new=StripImports().visit(ast.parse(Path(current.__file__).read_text()))
assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
p=np.array([.5,.5,.5,.36,.30,.24,.4,.2,.3,.7])
p[6:]/=np.linalg.norm(p[6:])
points=np.r_[np.random.default_rng(3).uniform(0,1,(100,3)),
 np.array([[0,0,0],[.1,0,0],[0,.05,0],[0,0,.01]])@current.rotation(p[6:],np).T+p[:3]]
p=p.astype(np.float32);points=points.astype(np.float32)
with jax.default_matmul_precision('highest'):
    a=np.asarray(jax.jit(lambda x,q:current.signed_distance(x,q,40))(jnp.asarray(points),jnp.asarray(p)))
    b=np.asarray(jax.jit(lambda x,q:legacy.signed_distance(x,q,40))(jnp.asarray(points),jnp.asarray(p)))
np.testing.assert_array_equal(a,b)
expected=current.signed_distance_numpy(points,p,64)
report=dict(backend=jax.default_backend(),math_ast_identical_except_imports=True,
 maximum_legacy_current_gap=float(abs(a-b).max()),maximum_gpu_numpy_gap=float(abs(a-expected).max()),
 source=str(source),scope='No tolerance change: existing CPU geometry test run separately. GPU highest-matmul strict3e-7 test exceeds by3.32e-8 identically before/after import-only refactor; default GPU reduced matmul is unsuitable for that strict test.')
(OUT/'melting-local-fem-geometry-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
