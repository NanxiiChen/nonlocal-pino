"""Explicit research experiment identities, separate from numerical configuration."""


def capacity_case(name):
    if name == 'width192':
        return dict(control='capacity', width=192, features=96, parameters=2021185, iterations=30000,
            baseline_width=128, baseline_features=64, baseline_parameters=905089, baseline_cap=256,
            config='examples.confined_ch2d.configs.train_operator_smallscale10_512_width192_fp32',
            receipt='confined-ch-width192', token='-confined-width192-fp32-',
            prior_receipt='confined-ch-fp32')
    if name == 'width256_50k':
        return dict(control='capacity', width=256, features=128, parameters=3579649, iterations=50000,
            baseline_width=192, baseline_features=96, baseline_parameters=2021185, baseline_cap=512,
            config='examples.confined_ch2d.configs.train_operator_smallscale10_512_width256_fp32_50k',
            receipt='confined-ch-width256-50k', token='-confined-width256-fp32-50k-',
            prior_receipt='confined-ch-width192')
    if name == 'width192_bank1024_50k':
        return dict(control='bank_growth', width=192, features=96, parameters=2021185, iterations=50000,
            baseline_width=192, baseline_features=96, baseline_parameters=2021185, baseline_cap=512,
            config='examples.confined_ch2d.configs.train_operator_smallscale10_1024_width192_fp32_50k',
            receipt='confined-ch-width192-bank1024-50k', token='-confined-width192-bank1024-fp32-50k-',
            prior_receipt='confined-ch-width192',
            dense_reference='/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_dense_validation_1024')
    raise ValueError(f'Unknown explicit capacity case: {name}')
