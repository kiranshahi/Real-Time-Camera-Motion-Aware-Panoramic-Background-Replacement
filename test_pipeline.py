import unittest
import numpy as np
from panovideo.geometry import PanoramaRenderer, rotation_matrix
from panovideo.segmenter import composite, refine_mask

class GeometryTests(unittest.TestCase):
    def setUp(self):
        h,w = 200,400
        xx=np.tile(np.arange(w,dtype=np.uint8),(h,1))
        self.panorama=np.dstack([xx,np.zeros_like(xx),np.full_like(xx,70)])
        self.renderer=PanoramaRenderer(self.panorama,80,40,70)

    def test_rotation_matrix_orthonormal(self):
        R=rotation_matrix(17,29,7)
        self.assertTrue(np.allclose(R.T@R,np.eye(3),atol=1e-6))

    def test_render_dimensions(self):
        self.assertEqual(self.renderer.render().shape,(40,80,3))

    def test_yaw_changes_panorama(self):
        a=self.renderer.render(0)
        b=self.renderer.render(30)
        self.assertFalse(np.array_equal(a,b))

    def test_compositing(self):
        foreground=np.full((3,4,3),200,dtype=np.uint8)
        background=np.full((3,4,3),20,dtype=np.uint8)
        self.assertTrue(np.array_equal(composite(foreground,background,np.ones((3,4))),foreground))
        self.assertTrue(np.array_equal(composite(foreground,background,np.zeros((3,4))),background))

    def test_mask_refinement(self):
        m=np.zeros((30,30),dtype=np.float32);m[10:20,10:20]=1
        result=refine_mask(m)
        self.assertEqual(result.shape,m.shape)
        self.assertTrue(np.min(result)>=0 and np.max(result)<=1)

if __name__=='__main__':
    unittest.main()
