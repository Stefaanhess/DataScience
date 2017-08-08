import os
class Settings:
    def __init__(self):
        # Couzin:
        self.Rr = 20
        self.Ro = 80
        self.Ra = 200
        self.max_turn = 90
        self.norm = 3

        # Simulation:
        self.N = 20
        self.winWidth = 500
        self.winHeight = 500
        self.vision_radius = self.Ra
        self.vision_bins = 36
        self.outfile = 'Tracks/Tracks_{}_{}_{}.npy'.format(self.Rr, self.Ro, self.Ra)
        # Network:
        self.folder = 'Graphs/{}_{}_{}/'.format(self.Rr, self.Ro, self.Ra)
        self.min_track_length = 2
        self.num_discretization_bins = 36

        if not os.path.exists(self.folder):
            os.mkdir(self.folder)