import shutil

from .data import p
from cultionet.scripts.cultionet import open_config
from cultionet.data.create import create_predict_dataset
from cultionet.utils import model_preprocessing
from cultionet.utils.project_paths import setup_paths


CONFIG = open_config(p / 'config.yml')
END_YEAR = CONFIG['years'][-1]
REGION = f"{CONFIG['regions'][-1]:06d}"


def get_image_list():
    image_list = []
    for image_vi in CONFIG['image_vis']:
        vi_path = p / 'time_series_vars' / REGION / image_vi
        ts_list = model_preprocessing.get_time_series_list(
            vi_path, END_YEAR-1, CONFIG['start_date'], CONFIG['end_date'], date_format='%Y%j'
        )
        image_list += ts_list

    return image_list


def test_predict_dataset():
    ppaths = setup_paths('.', append_ts=True)
    image_list = get_image_list()

    create_predict_dataset(
        image_list=image_list,
        region=REGION,
        year=END_YEAR,
        process_path=ppaths.get_process_path('predict'),
        gain=1e-4,
        offset=0.0,
        ref_res=10.0,
        resampling='nearest',
        window_size=50,
        padding=5,
        num_workers=2,
        chunksize=100
    )
    pt_list = list(ppaths.get_process_path('predict').glob('*.pt'))

    assert len(pt_list) > 0, 'No .pt files were created.'

    shutil.rmtree(str(ppaths.get_process_path('predict')))
