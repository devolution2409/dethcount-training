"""TODO(dark_souls_remastered): Add a description here."""

from __future__ import absolute_import, division, print_function

import itertools
import os

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

# TODO(dark_souls_remastered): BibTeX citation
_CITATION = """
"""

# TODO(dark_souls_remastered):
_DESCRIPTION = """Dataset to recognize Dark Souls Remastered death frames. It contains 720p 'YOU DIED' screenshots as well as regular gameplay.
"""

SUPPORTED_IMAGE_FORMAT = (".jpg", ".jpeg", ".png")

class DarkSoulsRemastered(tfds.core.GeneratorBasedBuilder):
  """TODO(dark_souls_remastered): Short description of my dataset."""

  # TODO(dark_souls_remastered): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # TODO(dark_souls_remastered): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(),
            # Here, labels can be of 2 distinct values, ALIVE or DEAD
            "label": tfds.features.ClassLabel(num_classes=2),
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=("image","label "),
        # Homepage of the dataset for documentation
        #homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators from the folder names."""
    # At data creation time, parse the folder to deduce number of splits,
    # labels, image size,

    # The splits correspond to the high level folders
    split_names = list_folders(dl_manager.manual_dir)

    # Extract all label names and associated images
    split_label_images = {}  # dict[split_name][label_name] = list(img_paths)
    for split_name in split_names:
      split_dir = os.path.join(dl_manager.manual_dir, split_name)
      split_label_images[split_name] = {
          label_name: list_imgs(os.path.join(split_dir, label_name))
          for label_name in list_folders(split_dir)
      }

    # Merge all label names from all splits to get the final list of labels
    # Sorted list for determinism
    labels = [split.keys() for split in split_label_images.values()]
    labels = list(sorted(set(itertools.chain(*labels))))

    # Could improve the automated encoding format detection
    # Extract the list of all image paths
    image_paths = [
        image_paths
        for label_images in split_label_images.values()
        for image_paths in label_images.values()
    ]
    if any(f.lower().endswith(".png") for f in itertools.chain(*image_paths)):
      encoding_format = "png"
    else:
      encoding_format = "jpeg"

    # Update the info.features. Those info will be automatically resored when
    # the dataset is re-created
    self.info.features["image"].set_encoding_format(encoding_format)
    self.info.features["label"].names = labels

    def num_examples(label_images):
      return sum(len(imgs) for imgs in label_images.values())

    # Define the splits
    return [
        tfds.core.SplitGenerator(
            name=split_name,
            # The number of shards is a dynamic function of the total
            # number of images (between 0-10)
            num_shards=min(10, max(num_examples(label_images) // 1000, 1)),
            gen_kwargs=dict(label_images=label_images,),
        ) for split_name, label_images in split_label_images.items()
    ]

  def _generate_examples(self):
    """Yields examples."""
    # TODO(dark_souls_remastered): Yields (key, example) tuples from the dataset
    yield 'key', {}


def list_folders(root_dir):
  return [
    f for f in tf.io.gfile.listdir(root_dir)
      if tf.io.gfile.isdir(os.path.join(root_dir, f))
  ]
  
def list_imgs(root_dir):
  return [
    os.path.join(root_dir, f)
    for f in tf.io.gfile.listdir(root_dir)
      if any(f.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMAT)
]
