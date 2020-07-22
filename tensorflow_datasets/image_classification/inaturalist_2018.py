# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""INaturalist datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import six.moves.urllib as urllib
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_CITATION = """\
@InProceedings{Horn_2018_CVPR,
author = {
Van Horn, Grant and Mac Aodha, Oisin and Song, Yang and Cui, Yin and Sun, Chen
and Shepard, Alex and Adam, Hartwig and Perona, Pietro and Belongie, Serge},
title = {The INaturalist Species Classification and Detection Dataset},
booktitle = {
The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
"""

_URL_2018 = "http://www.vision.caltech.edu/~gvanhorn/datasets/inaturalist/fgvc5_competition/"



_2018DESCRIPTION = """\
This dataset contains a total of 8412 categories, across 437513 training
images and 24436 validation images. For the training set, the distribution of
images per category follows the observation frequency of that category by the
iNaturalist community.

Although the original dataset contains some images with bounding boxes,
currently, only image-level annotations are provided (single label/image).
In addition, the organizers have not published the test labels, so we only
provide the test images (label = -1).
"""
class INaturalist2018(tfds.core.GeneratorBasedBuilder):
  """Dataset from the INaturalist Competition 2018."""

  VERSION = tfds.core.Version("0.1.0")

  def _info(self):
    """Define the dataset info."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_2018DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "id": tfds.features.Text(),
            "image": tfds.features.Image(),
            "label": tfds.features.ClassLabel(
               num_classes=8412),
            "supercategory": tfds.features.ClassLabel(
                names_file=tfds.core.get_tfds_path(
                    os.path.join(
                        "image_classification",
                        "inaturalist_supercategories.txt"))),
        }),
        supervised_keys=("image", "label"),
        homepage="https://github.com/visipedia/inat_comp/tree/master/2018",
        citation=_CITATION)

  def _split_generators(self, dl_manager):
    output_files = dl_manager.download_and_extract({
        "train_val2018":
            tfds.download.Resource(
                url=urllib.parse.urljoin(_URL_2018, "train_val2018.tar.gz"),
                extract_method=tfds.download.ExtractMethod.NO_EXTRACT),
        "train_annos":
            tfds.download.Resource(
                url=urllib.parse.urljoin(_URL_2018, "train2018.json.tar.gz"),
                extract_method=tfds.download.ExtractMethod.NO_EXTRACT),
        "val_annos":
            tfds.download.Resource(
                url=urllib.parse.urljoin(_URL_2018, "val2018.json.tar.gz"),
                extract_method=tfds.download.ExtractMethod.NO_EXTRACT),
        "test_2018":
            tfds.download.Resource(
                url=urllib.parse.urljoin(_URL_2018, "test2018.tar.gz"),
                extract_method=tfds.download.ExtractMethod.NO_EXTRACT),

    })
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs=dict(
                images_archive=dl_manager.iter_archive(
                    output_files["train_val2018"]),
                annon_file=os.path.join(output_files["train_annos"],
                                        "train2018.json"),
            ),
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs=dict(
                images_archive=dl_manager.iter_archive(
                    output_files["train_val2018"]),
                annon_file=os.path.join(output_files["val_annos"],
                                        "val2018.json"),
            ),
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs=dict(
                images_archive=dl_manager.iter_archive(
                    output_files["test2018"]),
                annon_file=None,
            ),
        ),
    ]

  def _generate_examples(self, images_archive, annon_file):
    """Generate examples."""
    if annon_file is not None:
      # Training and validation images.
      with tf.io.gfile.GFile(annon_file, "r") as f:
        data = json.load(f)
      # First read the annotations file, used to filter the contents of the
      # tar.gz file when yielding examples.
      key2data = {}
      for image, annotation in zip(data["images"], data["annotations"]):
        category_id = annotation["category_id"]
        category = data["categories"][category_id]["name"]
        supercategory = data["categories"][category_id]["supercategory"]
        key = os.path.basename(image["file_name"]).split(".")[0]
        key2data[key] = {
            "id": key,
            "label": category,
            "supercategory": supercategory,
        }
      # Read tar.gz file containing train & validation images and yield relevant
      # examples.
      for fpath, fobj in images_archive:
        key = os.path.basename(fpath).split(".")[0]
        if key in key2data:
          data = key2data[key].copy()
          data["image"] = fobj
          yield key, data
    else:
      # Read tar.gz file containing all test images and yield all examples.
      for fpath, fobj in images_archive:
        key = os.path.basename(fpath).split(".")[0]
        # Note: test labels are not annotated, so just return -1 as labels.
        yield key, {
            "id": key,
            "image": fobj,
            "label": -1,
            "supercategory": -1,
        }
