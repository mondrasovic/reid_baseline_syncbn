import click
import os
import sys
import tqdm
import shutil
import pathlib
import itertools
import collections

import cv2 as cv

from xml.etree import ElementTree


def iter_frame_targets(frame):
    def _coord(x):
        return int(round(float(x)))

    for target in frame.findall('.//target'):
        obj_id = int(target.attrib['id'])

        box_attr = target.find('box').attrib
        x = _coord(box_attr['left'])
        y = _coord(box_attr['top'])
        w = _coord(box_attr['width'])
        h = _coord(box_attr['height'])
        bbox = (x, y, w, h)

        yield obj_id, bbox


def iter_track_sample_content(xml_file_path, subset_type):
    tree = ElementTree.parse(xml_file_path)
    root = tree.getroot()

    sample_name = root.attrib['name']

    data_dir_path = 'Insight-MVT_Annotation_' + subset_type.capitalize()
    imgs_dir_path = os.path.join(data_dir_path, sample_name)

    for frame in root.findall('./frame'):
        frame_num = int(frame.attrib['num'])
        img_file_name = f'img{frame_num:05d}.jpg'
        img_file_path = os.path.join(imgs_dir_path, img_file_name)
        targets_iter = iter_frame_targets(frame)

        yield img_file_path, targets_iter



class VeRiDatasetWriter():
    def __init__(
        self,
        dataset_dir_path,
        output_dir_path,
        start_obj_id=0,
        start_img_id=0,
        context=0.0,
        save_kth_occurr=1
    ) -> None:
        assert start_obj_id >= 0, "object ID must be non-negative"
        assert start_img_id >= 0, "image ID must be non-negative"
        assert save_kth_occurr > 0, "k-th occurence must be positive"

        self.dataset_dir_path = dataset_dir_path
        self.output_dir_path = output_dir_path

        self.obj_id_iter = itertools.count(start=start_obj_id)
        self.img_id_iter = itertools.count(start=start_img_id)

        self.context = context
        self.save_kth_occurr = save_kth_occurr

        if os.path.exists(self.output_dir_path):
            shutil.rmtree(self.output_dir_path)
        os.makedirs(self.output_dir_path, exist_ok=True)
    
    def process_sample(self, sample_iter):
        obj_id_map = {}
        occurr_iter_map = collections.defaultdict(lambda: itertools.count())

        for img_rel_file_path, targets_iter in sample_iter:
            img_file_path = os.path.join(
                self.dataset_dir_path, img_rel_file_path
            )
            img = cv.imread(img_file_path, cv.IMREAD_COLOR)

            for obj_id, bbox in targets_iter:
                obj_occurr_id = next(occurr_iter_map[obj_id])
                if obj_occurr_id % self.save_kth_occurr != 0:
                    continue

                glob_obj_id = obj_id_map.get(obj_id)
                if not glob_obj_id:
                    glob_obj_id = next(self.obj_id_iter)
                    obj_id_map[obj_id] = glob_obj_id
 
                cam_id = obj_occurr_id
                img_id = next(self.img_id_iter)
                
                file_name = f'{glob_obj_id:04d}_c{cam_id:03d}_{img_id:07d}.jpg'
                output_img_file_path = os.path.join(
                    self.output_dir_path, file_name
                )
                roi = self._extract_bbox(img, bbox)
                
                cv.imwrite(output_img_file_path, roi)
    
    def _extract_bbox(self, img, bbox):
        x, y, w, h = bbox
        w = int(round(w * (1 + self.context)))
        h = int(round(h * (1 + self.context)))
        roi = img[y:y + h, x:x + w]

        return roi


@click.command()
@click.argument('input_dir_path', type=click.Path(exists=True))
@click.argument('output_dir_path', type=click.Path())
@click.option(
    '-s', '--start-obj-id', type=int, default=0, show_default=True,
    help="Initial object ID (unique within dataset)."
)
@click.option(
    '-S', '--start-img-id', type=int, default=0, show_default=True,
    help="Initial image ID (unique within dataset)."
)
@click.option(
    '-c', '--context', type=float, default=0.25, show_default=True,
    help="Bounding box context."
)
@click.option(
    '-k', '--save-kth-occurr', type=int, default=3, show_default=True,
    help="Save only every k-th occurrence of a specific object."
)
@click.option(
    '--subset-type', type=click.Choice(['train', 'test']), default='train',
    show_default=True, help="Data subset type."
)
def main(
    input_dir_path,
    output_dir_path,
    start_obj_id,
    start_img_id,
    context,
    subset_type,
    save_kth_occurr
):
    """Creates a VeRi-like dataset for vehicle re-identification (ReID) based on
    UA-DETRAC multiple object tracking dataset.
    """
    dataset_writer = VeRiDatasetWriter(
        input_dir_path, output_dir_path, start_obj_id, start_img_id, context,
        save_kth_occurr
    )

    anno_dir_name = '540p-' + subset_type.capitalize()
    anno_dir = pathlib.Path(input_dir_path) / 'DETRAC_public' / anno_dir_name

    tqdm_pbar = tqdm.tqdm(file=sys.stdout)
    with tqdm_pbar as pbar:
        for xml_file in anno_dir.glob("*.xml"):
            pbar.set_description(f"processing sample {xml_file.stem}")
            sample_iter = iter_track_sample_content(str(xml_file), subset_type)
            dataset_writer.process_sample(sample_iter)
            pbar.update()

    return 0


if __name__ == '__main__':
    sys.exit(main())
