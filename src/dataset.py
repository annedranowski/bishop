from torchvision.datasets import VisionDataset
import os
import os.path
import sys

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename, extensions):
  return filename.lower().endswith(extensions)

def is_image_file(filename):
  return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
  images = []
  dir = os.path.expanduser(dir)
  if not ((extensions is None) ^ (is_valid_file is None)):
    raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
  if extensions is not None:
    def is_valid_file(x):
      return has_file_allowed_extension(x, extensions)
  for target in sorted(class_to_idx.keys()):
    d = os.path.join(dir, target)
    if not os.path.isdir(d):
      continue
    for root, _, fnames in sorted(os.walk(d)):
      for fname in sorted(fnames):
        path = os.path.join(root, fname)
        if is_valid_file(path):
          item = (path, class_to_idx[target])
          images.append(item)

  return images

def pil_loader(path):
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')

class KnotsImageFolder(VisionDataset):
  def __init__(self, root, loader, extensions=None, transform=None,
               target_transform=None, is_valid_file=None):
    super().__init__(root, transform=transform,
                     target_transform=target_transform)
    classes, class_to_idx = self._find_classes(self.root)
    samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
    if len(samples) == 0:
        raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                            "Supported extensions are: " + ",".join(extensions)))

    self.loader = loader
    self.extensions = extensions

    self.classes = classes
    self.class_to_idx = class_to_idx
    self.samples = samples
    self.targets = [s[1] for s in samples]

  def _find_classes(self, dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()

    classes = [int(i) for i in classes]
    classes = [str(i) for i in range(0, max(classes)+1)]

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

  def __getitem__(self, index):
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)

    return sample, target

  def __len__(self):
    return len(self.samples)

class CustomImageFolder(CustomDatasetFolder):
  def __init__(self, root, transform=None, target_transform=None,
               loader=pil_loader, is_valid_file=None):
    super().__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
    self.imgs = self.samples
