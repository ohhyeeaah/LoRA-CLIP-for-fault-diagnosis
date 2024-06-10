import os

from .utils import Datum, DatasetBase
from .split_data import Split_Data


template = ['a photo of a {} machine.',
            'a photo of a {}.',
            'It is a {} machine',
            'a machine of {} state',
            'a photo of a {},a kind of machine state',
            'the photo of {},a kind of machine state']


class HUST_Gearbox(DatasetBase):

    dataset_dir = 'HUST-gearbox-dataset'

    def __init__(self, root, num_shots, working_condition):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'hust_gearbox_image/{}'.format(working_condition))
        self.split_path = os.path.join(self.dataset_dir, 'hust_gearbox_image/{}/split_hust_gearbox_{}.json'.format(working_condition, working_condition))

        self.template = template

        train, val, test = Split_Data.read_split(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)