import unittest
import numpy as np
from helical_fiber import Config, HelicalFiberModel, calibrate_force
class ModelTests(unittest.TestCase):
    def setUp(self):
        self.m=HelicalFiberModel(); self.s=np.linspace(0,4.277,6395)
    def test_independent_equation(self):
        s=self.s
        psi=np.sin(np.pi*8*s/4.277)*(-1.)**(np.floor(s/(4.277/8))+1)
        y=-.012*2.7*.017*psi*np.sin(.999*s/np.hypot(.017,.005/(2*np.pi))-.83)
        r=self.m.fit(s,y,method='ols',fit_offset=False)['summary']
        self.assertAlmostEqual(r['kappa_per_m'],2.7,places=9)
        self.assertAlmostEqual(r['phi_rad'],-.83,places=9)
    def test_contacts_and_robustness(self):
        events=[{'s_m':1.6,'force_n':2.,'width_m':.015},{'s_m':2.65,'force_n':2.5,'width_m':.012,'sign':-1}]
        y=self.m.simulate(self.s,2.5,.7,events,noise_std=.2e-6)['strain']
        r=self.m.fit(self.s,y); ols=self.m.fit(self.s,y,method='ols')
        self.assertLess(abs(r['summary']['kappa_per_m']-2.5),.002)
        self.assertLess(abs(r['summary']['phi_rad']-.7),.002)
        self.assertLess(abs(r['summary']['kappa_per_m']-2.5),abs(ols['summary']['kappa_per_m']-2.5))
        self.assertEqual(len(r['contacts']),2)
        for a,b in zip(r['contacts'],events):
            self.assertLess(abs(a['s_m']-b['s_m']),.002)
            self.assertLess(abs(a['force_n']-b['force_n']),.05)
    def test_zero(self):
        r=self.m.fit(self.s,np.zeros(len(self.s)))
        self.assertIsNone(r['summary']['phi_deg']); self.assertEqual(r['contacts'],[])
        xyz=self.m.helix_xyz(self.s)
        np.testing.assert_allclose(np.hypot(xyz[:,0],xyz[:,1]),.017)
        self.assertAlmostEqual(xyz[-1,2],.2)
        np.testing.assert_allclose(self.m.helix_xyz(self.s,1e-9,.3),xyz,atol=1e-9)
    def test_missing_and_offset(self):
        y=self.m.bending(self.s,1.3,-3.13,offset=20e-6); y[100:170]=np.nan; y[300]=np.inf
        r=self.m.fit(self.s,y)
        self.assertAlmostEqual(r['summary']['kappa_per_m'],1.3,places=8)
        self.assertAlmostEqual(r['summary']['phi_rad'],-3.13,places=8)
        self.assertAlmostEqual(r['summary']['offset_strain'],20e-6,places=10)
        self.assertEqual(r['summary']['valid_samples'],len(y)-71)
    def test_invalid(self):
        for s,y in [(self.s[::-1],self.s),(self.s,np.full(len(self.s),np.nan))]:
            with self.assertRaises(ValueError): self.m.fit(s,y)
        with self.assertRaises(ValueError): Config(strain_scale=0)
        with self.assertRaises(ValueError): self.m.coordinates([-1])
        with self.assertRaises(ValueError): self.m.fit(np.zeros(20),np.zeros(20))
    def test_calibration(self):
        f=np.array([1.,1.5,2.,2.5,3.]); r=calibrate_force(f,27.914e-6*f+35.526e-6)
        self.assertAlmostEqual(r['slope_strain_per_n'],27.914e-6)
        self.assertAlmostEqual(r['intercept_strain'],35.526e-6)
        self.assertEqual(self.m.zone(.02/.2*4.277),'disk')
        self.assertEqual(self.m.zone(.01/.2*4.277),'gap')
    def test_phase_and_nonuniform(self):
        s=np.sort(np.random.default_rng(3).uniform(0,4.277,4000))
        for phi in (-3.14,-1.,0.,2.,3.14):
            r=self.m.fit(s,self.m.bending(s,4.,phi))['summary']
            self.assertAlmostEqual(r['phi_rad'],phi,places=8)
            self.assertAlmostEqual(r['theta_deg'],np.rad2deg(.8),places=8)
    def test_zero_force(self):
        r=self.m.simulate(self.s,0,0,[{'s_m':1.,'force_n':0.,'width_m':.01}],noise_std=0)
        np.testing.assert_array_equal(r['strain'],0.)
if __name__=='__main__': unittest.main()
