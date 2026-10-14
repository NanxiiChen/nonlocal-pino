"""Independent TensorBoard consumer verifies physical endpoint images and steps."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from matplotlib.tri import Triangulation
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from examples.fracture2d.tensorboard_monitor import endpoint_fields,publish_endpoint,sha


class FractureTensorBoardTest(unittest.TestCase):
    def test_completed_load_images_use_matched_reference_and_correct_step(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'pinn';folder=root/'state_0053';folder.mkdir(parents=True)
            refroot=base/'reference';ref=(refroot/'state_0056');ref.mkdir(parents=True)
            values=dict(d=np.array([.2,.3,.4]),H=np.array([.1,.2,.3]),
                q=np.array([-.01,0.,0.,.0053,.01,.0053]))
            np.savez(folder/'fields.npz',**values)
            np.savez(ref/'fields.npz',**values)
            row=dict(index=53,load=.0053,reaction=2.,damage_max=.4,physical_AT2=.2,
                field_stable=False,certified_path=False,q_d_synchronized=True,
                fields_sha256=sha(folder/'fields.npz'))
            xy=np.array([[0.,0.],[1.,0.],[0.,1.]])
            writer=SummaryWriter(str(base/'events'))
            record=publish_endpoint(writer,root,root/'figures',Triangulation(xy[:,0],xy[:,1],[[0,1,2]]),
                xy,row,[row],refroot,[dict(index=56,load=.0053,reaction=2.)],
                dpi=40,dofs=6,volumes=np.ones(3))
            writer.close()
            accumulator=EventAccumulator(str(base/'events'));accumulator.Reload()
            expected=['train/damage_field','train/ux_field','train/uy_field',
                      'train/history_field','train/load_reaction_curve']
            self.assertEqual(set(accumulator.Tags()['images']),set(expected))
            for tag in expected:self.assertEqual(accumulator.Images(tag)[0].step,53)
            self.assertEqual(accumulator.Scalars('train/reaction')[0].value,2.)
            self.assertEqual(record['relative_damage_L2'],0.)
            self.assertEqual(record['reference_load'],.0053)
            with self.assertRaises(ValueError):
                endpoint_fields(folder/'fields.npz',dict(row,q_d_synchronized=False),nodes=3,dofs=6)


if __name__=='__main__':unittest.main()
