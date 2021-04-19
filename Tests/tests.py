import pytest
import unittest
pytest .__path__

import threebody
from threebody.EDO.EDO_3 import distance

#A python module is a set of python functions and statements that can be loaded.
#L'objectif de ce module est d'effectuer des tests sur nos programmes

class test_edo3(unittest.TestCase):


    def test_distance(self):
       """"vvhgfg""""
        
        X = np.array([0, 1])
        Y = np.array([0, 2])
        distance(X,Y) == 1
        

    def trajectories(self):


        x_pnts1, y_pnts1, x_pnts2, y_pnts2, x_pnts3, y_pnts3, z_pnts1, z_pnts2, z_pnts3 = trajectories(h=100)
        size = len(x_pnts1)
        self.assertEqual(len(x_pnts2), len(x_pnts3), 
                        len(y_pnts1), len(y_pnts2), len(y_pnts3), 
                        len(z_pnts1), len(z_pnts2), len(z_pnts3), size)