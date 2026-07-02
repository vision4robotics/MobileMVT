import numpy as np

class GMCLoader:
    def __init__(self, gmc_file):
        self.gmc_file = gmc_file
        self.affines = dict()
        self.load_gmc()

    def load_gmc(self):
        with open(self.gmc_file, 'r') as f:
            for line in f.readlines():
                line = line.strip().split()
                frame_id = int(line[0]) + 1

                affine = np.zeros((2, 3))
                affine[0, 0] = float(line[1])
                affine[0, 1] = float(line[2])
                affine[0, 2] = float(line[3])
                affine[1, 0] = float(line[4])
                affine[1, 1] = float(line[5])
                affine[1, 2] = float(line[6])

                self.affines[frame_id] = affine


    def get_affine(self,frame_id):
        return self.affines[frame_id]