"""The finer fixed-particle study must reject hidden scheme changes."""
import copy
import unittest
from examples.electropolish2d.configs.train_operator import get_config
from docs.tasks.electro_horizon_ratio_study import fixed_particle_config
from scripts.render_electro_horizon_study import fixed_config


class RatioControlTests(unittest.TestCase):
    def test_allow_horizon_and_capacity_but_keep_grid_and_strategy(self):
        base=get_config().to_dict()
        other=copy.deepcopy(base)
        other['nl']['delta']=.0075
        other['sample']['spatial']['max_nodes']+=512
        other['sample']['spatial']['max_edges']+=512
        other['log']['run_name']='new-control'
        other['log']['log_dir']='/tmp/new-control'
        self.assertEqual(fixed_particle_config(base),fixed_particle_config(other))
        for section,name,value in [('sample','Nx',201),('sample','Nt',8),
                                   ('train','iterations',5000),('physical','L',2e-6)]:
            changed=copy.deepcopy(other);changed[section][name]=value
            self.assertNotEqual(fixed_particle_config(base),fixed_particle_config(changed))
        self.assertEqual(base,get_config().to_dict())

    def test_legacy_density_policy_still_only_allows_its_old_differences(self):
        base=get_config().to_dict();other=copy.deepcopy(base)
        other['sample']['Nx']=201;other['sample']['Ny']=101
        self.assertEqual(fixed_config(base),fixed_config(other))
        other['log']['log_dir']='/tmp/different-root'
        self.assertNotEqual(fixed_config(base),fixed_config(other))


if __name__=='__main__':
    unittest.main()
