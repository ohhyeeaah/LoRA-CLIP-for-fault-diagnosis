from .cwru import CWRU
from .hust_bearing import HUST_Bearing
from .hust_gearbox import HUST_Gearbox
from .lw import LW
from .xj import XJ
from .pu import PU



dataset_list = {
                "cwru": CWRU,
                "hust_bearing": HUST_Bearing,
                "hust_gearbox": HUST_Gearbox,
                "lw": LW,
                "xj": XJ,
                "pu": PU
                }


def build_dataset(dataset, root_path, shots, working_condition):
    return dataset_list[dataset](root_path, shots, working_condition)