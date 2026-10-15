"""Direct bounded-head loading; extra same-head adaptation, no screening gate."""
from examples.fracture2d._legacy.configs.bounded_damage_morphology_trial import get_config as base


def get_config():
    c = base()
    a = c.head
    a.mode = 'direct_loading'
    a.output_dir = '/root/autodl-tmp/tf-logs/fracture2d/bounded_damage_loading_20261013'
    a.heads = ['sigmoid', 'exp_square']
    a.fixed_indices = [52]
    a.fixed_iterations = 8192
    a.lbfgs.maxiter = a.fixed_iterations
    a.warm_root = '/root/autodl-tmp/tf-logs/fracture2d/bounded_damage_morphology_20261013'
    digests = {
        'sigmoid/network.eqx': '4a79c8cb8fd475d36fa2850d701c988b3e2971ba27d3123c690576232391628a',
        'sigmoid/parameterization.npz': '11b29d7021c364a809dc3b9958114e04fc4f58db73c85c0d32ec84ebcd69466e',
        'sigmoid/result.json': 'b70531866ac005a48c4eb296fc929e2d9727fb6d372411453456b4fc4b4fe3c5',
        'sigmoid/fields.npz': 'dea6d0e4695a02b4cb7f2b213a416d0fe90f3f3720fb3a3e69c67089437c82f0',
        'exp_square/network.eqx': '13098194ee7a88686c963ff7dd89d2bd492a8cd842473831a7ea19e00c0c8744',
        'exp_square/parameterization.npz': '3c91303e23a587148d2758fe2f9170710d75028ce0fdbfc52cc0c868891d92cd',
        'exp_square/result.json': '87ca4872c13fad2d908e992dcf2db3a12bb1f7e57fb06c230a2d457bd6505bc0',
        'exp_square/fields.npz': '532837c86a275270dbfc97599ca32de4a7146686826d7f135b370b401a59c970',
    }
    a.warm_pins = [dict(path=a.warm_root+'/fixed_0052/'+name,sha256=digest) for name,digest in digests.items()]
    a.warm_pins.append(dict(path=a.warm_root+'/identity.json',
        sha256='f6030b362b8f62d1040fd9cbe33093f2406960d816fb47d36ab19ebdfe737073'))
    return c
