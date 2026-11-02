"""Frozen independent validation parameters and strict family/protocol checks."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from examples.channel_nsch2d.configs.train_operator_conservative_online import get_config
from examples.channel_nsch2d.cohort import prepare,load,reserved_amplitudes
from examples.channel_nsch2d.online import sample_geometry
from examples.channel_nsch2d.cohort_evaluate import field_metrics,response_metric


class ChannelCohort(unittest.TestCase):
    def test_offgrid_layout_avoids_training_nodes(self):
        from scipy.spatial import cKDTree
        from examples.channel_nsch2d.problem import quadrature,walls
        c=get_config();a=np.array([.12,.08]);points,_=quadrature(a,c,192,80)
        x,e=np.meshgrid((np.arange(101)+.5)*4/101,(np.arange(43)+.5)/43,indexing='ij')
        b,t=walls(x,a,c,xp=np);query=np.stack((x,b+e*(t-b)),axis=-1).reshape(-1,2)
        distance,_=cKDTree(points).query(query)
        self.assertEqual(int(np.sum(distance<1e-12)),1)

    def test_metrics_expose_minority_failure_and_ignore_unprescribed_initial_pressure(self):
        target=np.ones((3,501,5));target[...,0]=1;target[:,0,0]=-1
        predicted=target.copy();predicted[:,0,0]=1
        predicted[0,:,4]=100
        metrics=field_metrics(predicted,target,np.ones(501))
        self.assertLess(metrics['phi'],.1)
        self.assertEqual(metrics['dispersed_fraction'],1.)
        self.assertEqual(metrics['pressure'],0.)
        self.assertEqual(metrics['velocity'],0.)
        target=np.ones((3,4,2))+np.array([-.1,0,.1])[:,None,None]
        blind=np.broadcast_to(target.mean(axis=0),target.shape)
        response=response_metric(blind,target)
        self.assertAlmostEqual(response['relative_error'],1.)
        self.assertEqual(response['amplitude_ratio'],0.)
        self.assertIsNone(response['correlation'])

    def test_frozen_prior_draws_are_independent_and_strictly_loaded(self):
        c=get_config();c.train.seed=42
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reserved'/'manifest.json';c.operator.validation_manifest=str(path)
            value=prepare(c,path,16,(101,43));self.assertEqual(value,load(path,c))
            a=reserved_amplitudes(c)
            initial=np.random.default_rng(c.operator.dataset_seed).uniform(.04,.14,(16,2))
            online=np.asarray([sample_geometry(c,i) for i in range(1000)])
            for other in (initial,online,np.array([[.12,.08]])):
                self.assertFalse(np.any(np.all(a[:,None,:].astype('float32')==other[None,:,:].astype('float32'),axis=-1)))
            with self.assertRaises(FileExistsError):prepare(c,path,16,(101,43))
            c.physical.mobility*=2
            with self.assertRaisesRegex(ValueError,'physical family'):load(path,c)
            c.physical.mobility/=2
            value['cases'][0]['amplitudes'][0]+=.001;path.write_text(json.dumps(value))
            with self.assertRaises(AssertionError):load(path,c)

    def test_malformed_protocol_and_case_identity_are_rejected(self):
        c=get_config()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reserved'/'manifest.json';value=prepare(c,path,2,(101,43))
            value['cases'][0]['draw_id']=1;path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'immutable draw'):load(path,c)
            value['cases'][0]['draw_id']=0;value['draw_namespace']+=1;path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'draw protocol'):load(path,c)


if __name__=='__main__':unittest.main()
