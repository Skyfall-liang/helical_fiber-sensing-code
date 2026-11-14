import unittest
import numpy as np
import torch
from fiber_net.data import generate
from fiber_net.physics import geometry,STRAIN_UNIT,angles
from fiber_net.network import JointModel,loss_function
from fiber_net.theory_reference import HelicalFiberModel
class Tests(unittest.TestCase):
    def test_physics_matches_reference(self):
        s,x,_,_=geometry();coef=np.array([.4,-.3,0.])
        np.testing.assert_allclose(x@coef*STRAIN_UNIT,HelicalFiberModel().bending(s,.5,np.arctan2(-.3,.4)),atol=1e-12)
    def test_three_classes_and_force_decoder(self):
        a,e=generate(9,42);s,x,c,b=geometry()
        self.assertTrue(np.all(a['contact'][a['kind']==0]==0))
        self.assertTrue(np.all(a['coef'][a['kind']==2,:2]==0))
        np.testing.assert_allclose(a['contact'],a['support']*(c*a['force']+b),atol=2e-7)
        self.assertEqual(len(e[0]),0);self.assertGreater(len(e[1]),0)
    def test_reproducible_independent_splits(self):
        a,_=generate(3,12);b,_=generate(3,12);c,_=generate(3,13)
        np.testing.assert_array_equal(a['y'],b['y']);self.assertFalse(np.array_equal(a['y'],c['y']))
    def test_gradient_and_decoder(self):
        torch.set_num_threads(2);m=JointModel();a,_=generate(3,2)
        batch={k:torch.from_numpy(a[k]) for k in ('y','coef','contact','support','force')}
        out=m(batch['y']);loss,_=loss_function(out,batch);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for head in (m.force_head,m.support_head,m.shape_head[-1]):
            self.assertIsNotNone(head.weight.grad);self.assertTrue(torch.isfinite(head.weight.grad).all())
        torch.testing.assert_close(out['reconstruction'],out['bending']+out['contact'])
    def test_zero_phase_undefined(self):self.assertIsNone(angles([0,0,0])[1])
if __name__=='__main__':unittest.main()
