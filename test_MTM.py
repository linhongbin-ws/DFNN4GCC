import dvrk
import numpy as np

arm = dvrk.arm('MTMR')
arm.move_joint(np.array([0.1,0.1,0.1,0.,0.,0.,0.]))
