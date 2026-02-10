import os


class Parameters:
    def __init__(self):
        self.base_dir = '../antrenare'
        self.base_dir_valid ='../validare'
        self.dir_pos_examples_fred = os.path.join(self.base_dir, 'exempleFred')
        self.dir_pos_examples_velma = os.path.join(self.base_dir, 'exempleVelma')
        self.dir_pos_examples_daphne = os.path.join(self.base_dir, 'exempleDaphne')
        self.dir_pos_examples_shaggy = os.path.join(self.base_dir, 'exempleShaggy')
        self.dir_neg_examples = os.path.join(self.base_dir, 'exempleUnknown')
        self.dir_neg_examples_background = os.path.join(self.base_dir, 'exempleBackground')
        self.dir_test_examples = os.path.join(self.base_dir_valid,'validare')
        self.path_annotations = os.path.join(self.base_dir_valid, 'task1_gt_validare.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere2')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window_width = 64  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_window_height = 84
        self.overlap = 0.3
        self.number_positive_examples = 6547  # numarul exemplelor pozitive
        self.number_negative_examples = 40000  # numarul exemplelor negative
        self.has_annotations = False
        self.use_flip_images = True
        self.threshold = 0
        self.step = 8
