# import modules
import torch
import torchvision
from torchvision.models import resnet18, resnet50, ResNet50_Weights, ResNet18_Weights
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import tarfile
import os
import shutil
import re
import numpy as np
import cv2
from PIL import ImageFilter
from scipy import ndimage
import copy


# Set the seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call the set_seed function to apply the seed
set_seed()

tar_file = "./drive/MyDrive/DL_project/imagenet-a.tar"
data_folder = "imagenet-a"

# function to untar the dataset and store it in a new folder
def extract_dataset(compress_file, destination_folder):
  # function to change dir names to their words description
  def change_folders_names(readme_file, dataset_root):
    with open(readme_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Match lines containing WordNet IDs and descriptions
            match = re.match(r'n\d+ (.+)', line)
            if match:
                # Split the line into WordNet ID and description
                parts = match.group(0).split()
                wordnet_id = parts[0]
                description = ' '.join(parts[1:])
                os.rename(os.path.join(dataset_root, wordnet_id),
                            os.path.join(dataset_root, description))

  if not os.path.exists(compress_file):
    print("Compress file doesn't exist.")
    return

  if os.path.exists(destination_folder):
    # remove the folder if already exists one
    shutil.rmtree(destination_folder)

  # extract content from the .tar file
  with tarfile.open(compress_file, 'r') as tar_ref:
    tar_ref.extractall("./")
  print("All the data is extracted.")

  change_folders_names(destination_folder+"/README.txt", destination_folder)

extract_dataset(tar_file, data_folder)


from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# function that returns a DataLoader for the dataset
def get_data(batch_size, dataset_path, transform):

  data = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

  class_labels = data.classes
  print(f"The dataset contains {len(data)} images.")
  print(f"The dataset contains {len(class_labels)} labels.")

  test_loader = torch.utils.data.DataLoader(data, batch_size, shuffle=False, num_workers=4)
  # Create a subset with only the first 1000 images
  # subset_indices = list(range(32*10))  # Indices of the first 1000 images
  # data_subset = torch.utils.data.Subset(data, subset_indices)

  # # Create a DataLoader for the subset
  # test_loader = torch.utils.data.DataLoader(data_subset, batch_size=batch_size, shuffle=False, num_workers=8)
    

  return test_loader, class_labels


import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF

# function to display images from the DataLoader
def show_images(dataloader, num_images=5):
  # get a batch of data
  data_iter = iter(dataloader)
  images, labels = next(data_iter)

  # convert images to numpy array
  images = images.numpy()

  # display images
  fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
  for i in range(num_images):
      image = np.transpose(images[i], (1, 2, 0))  # move channels in last position
      image = np.clip(image, 0, 1)
      axes[i].imshow(image)
      axes[i].axis('off')
      axes[i].set_title(dataloader.dataset.classes[labels[i]])
  plt.show()


from PIL import ImageOps, ImageEnhance

def convert_to_pil(img):
    if isinstance(img, torch.Tensor):
        img = TF.to_pil_image(img)
    return img

def vertical_flip(img):
    img = convert_to_pil(img)
    res = img.transpose(Image.FLIP_TOP_BOTTOM)
    return TF.to_tensor(res)

def horizontal_flip(img):
    img = convert_to_pil(img)
    res = img.transpose(Image.FLIP_LEFT_RIGHT)
    return TF.to_tensor(res)

def brightness(img, factor_range=(0.5, 1.5)):
    img = convert_to_pil(img)
    factor = np.random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Brightness(img)
    res = enhancer.enhance(factor)
    return TF.to_tensor(res)

def color(img, factor_range=(0.5, 1.5)):
    img = convert_to_pil(img)
    factor = np.random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Color(img)
    res = enhancer.enhance(factor)
    return TF.to_tensor(res)

def sharpness(img, factor_range=(0.5, 1.5)):
    img = convert_to_pil(img)
    factor = np.random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Sharpness(img)
    res = enhancer.enhance(factor)
    return TF.to_tensor(res)

def rotation(img, angle_range=(-45, 45)):
    img = convert_to_pil(img)
    angle = np.random.uniform(angle_range[0], angle_range[1])
    res = img.rotate(angle)
    return TF.to_tensor(res)

def gaussian_blur(img, radius_range=(0.1, 2.0)):
    img = convert_to_pil(img)
    radius = np.random.uniform(radius_range[0], radius_range[1])
    res = img.filter(ImageFilter.GaussianBlur(radius))
    return TF.to_tensor(res)

def random_crop(img, size=(64, 64)):  # Default size provided
    img = convert_to_pil(img)
    width, height = img.size
    crop_width, crop_height = size
    if width < crop_width or height < crop_height:
        raise ValueError("Crop size must be less than the image size")
    left = np.random.randint(0, width - crop_width + 1)
    top = np.random.randint(0, height - crop_height + 1)
    right = left + crop_width
    bottom = top + crop_height
    res = img.crop((left, top, right, bottom))
    return TF.to_tensor(res)

def add_noise(img, mean=0, std=0.1):
    if isinstance(img, Image.Image):
        img = TF.to_tensor(img)
    noise = torch.randn(img.size(), device=img.device) * std + mean
    res = img + noise
    res = torch.clamp(res, 0, 1)
    return res

def affine_transform(img, max_translation=(10, 10), max_rotation=30, max_shear=10):
    img = convert_to_pil(img)
    width, height = img.size
    translation = (np.random.uniform(-max_translation[0], max_translation[0]),
                   np.random.uniform(-max_translation[1], max_translation[1]))
    rotation = np.random.uniform(-max_rotation, max_rotation)
    shear = np.random.uniform(-max_shear, max_shear)
    res = img.transform((width, height), Image.AFFINE, 
                        (1, shear, translation[0], shear, 1, translation[1]),
                        resample=Image.BILINEAR)
    res = res.rotate(rotation)
    return TF.to_tensor(res)

def elastic_deformation(img, alpha=34, sigma=4):
    img = convert_to_pil(img)
    img = np.array(img)
    random_state = np.random.RandomState(None)
    shape = img.shape
    dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    distored_image = ndimage.map_coordinates(img, indices, order=1, mode='reflect')
    return TF.to_tensor(distored_image.reshape(img.shape))

def contrast(img, factor_range=(0.5, 1.5)):
    img = convert_to_pil(img)
    factor = np.random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Contrast(img)
    res = enhancer.enhance(factor)
    return TF.to_tensor(res)

def saturation(img, factor_range=(0.5, 1.5)):
    img = convert_to_pil(img)
    factor = np.random.uniform(factor_range[0], factor_range[1])
    enhancer = ImageEnhance.Color(img)
    res = enhancer.enhance(factor)
    return TF.to_tensor(res)

def hue(img, factor_range=(-0.1, 0.1)):
    img = convert_to_pil(img)
    factor = np.random.uniform(factor_range[0], factor_range[1])
    res = TF.adjust_hue(img, factor)
    return TF.to_tensor(res)

def perspective_transform(img, distortion_scale=0.5):
    img = convert_to_pil(img)
    transform = torchvision.transforms.RandomPerspective(distortion_scale=distortion_scale, p=1.0)
    res = transform(img)
    return TF.to_tensor(res)

def channel_shuffle(img):
    if isinstance(img, Image.Image):
        img = TF.to_tensor(img)
    img = img[torch.randperm(3), :, :]
    return img

def random_erasing(img, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
    if isinstance(img, Image.Image):
        img = TF.to_tensor(img)
    c, h, w = img.size()
    area = h * w

    target_area = np.random.uniform(sl, sh) * area
    aspect_ratio = np.random.uniform(r1, 1/r1)

    h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
    w_erase = int(round(np.sqrt(target_area / aspect_ratio)))

    if h_erase > h:
        h_erase = h

    if w_erase > w:
        w_erase = w

    if h_erase > 0 and w_erase > 0:
        x1 = np.random.randint(0, h - h_erase + 1)
        y1 = np.random.randint(0, w - w_erase + 1)
        img[:, x1:x1 + h_erase, y1:y1 + w_erase] = torch.tensor(mean).view(-1, 1, 1)

    return img

def histogram_equalization(img):
    img = convert_to_pil(img)
    img = np.array(img)
    if len(img.shape) == 3:
        for i in range(3):
            img[:,:,i] = cv2.equalizeHist(img[:,:,i])
    else:
        img = cv2.equalizeHist(img)
    return TF.to_tensor(img)

augmentations = [vertical_flip, horizontal_flip, brightness, color, sharpness, rotation, gaussian_blur, lambda img: random_crop(img, size=(64, 64)), add_noise, affine_transform, elastic_deformation, contrast, saturation, hue, perspective_transform, channel_shuffle, random_erasing, histogram_equalization]



import random

# functon that apply B augmentations to the original image and return M+1 images
def augment_image(img, augmentations, B=15):
  assert len(augmentations) > 0, "There are not augmentations provided."

  images = [img]
  for _ in range(B):
    # randomly choose an augmentation in the augmentation functions
    index = random.randrange(0, len(augmentations))
    augmentation = augmentations[index]
    # apply the augmentation to the original image
    augmented_img = augmentation(img)
    # add the augmented image to the list of images I want to evaluate
    images.append(augmented_img)
  return images


# define the cost function used to evaluate the model output
def get_cost_function():
  cost_function = torch.nn.CrossEntropyLoss()
  return cost_function

# define the optimizer
def get_optimizer(net, lr, wd, momentum):
  optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
  return optimizer

thousand_k_to_200 = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: 0, 7: -1, 8: -1, 9: -1, 10: -1, 11: 1, 12: -1, 13: 2, 14: -1, 15: 3, 16: -1, 17: 4, 18: -1, 19: -1, 20: -1, 21: -1, 22: 5, 23: 6, 24: -1, 25: -1, 26: -1, 27: 7, 28: -1, 29: -1, 30: 8, 31: -1, 32: -1, 33: -1, 34: -1, 35: -1, 36: -1, 37: 9, 38: -1, 39: 10, 40: -1, 41: -1, 42: 11, 43: -1, 44: -1, 45: -1, 46: -1, 47: 12, 48: -1, 49: -1, 50: 13, 51: -1, 52: -1, 53: -1, 54: -1, 55: -1, 56: -1, 57: 14, 58: -1, 59: -1, 60: -1, 61: -1, 62: -1, 63: -1, 64: -1, 65: -1, 66: -1, 67: -1, 68: -1, 69: -1, 70: 15, 71: 16, 72: -1, 73: -1, 74: -1, 75: -1, 76: 17, 77: -1, 78: -1, 79: 18, 80: -1, 81: -1, 82: -1, 83: -1, 84: -1, 85: -1, 86: -1, 87: -1, 88: -1, 89: 19, 90: 20, 91: -1, 92: -1, 93: -1, 94: 21, 95: -1, 96: 22, 97: 23, 98: -1, 99: 24, 100: -1, 101: -1, 102: -1, 103: -1, 104: -1, 105: 25, 106: -1, 107: 26, 108: 27, 109: -1, 110: 28, 111: -1, 112: -1, 113: 29, 114: -1, 115: -1, 116: -1, 117: -1, 118: -1, 119: -1, 120: -1, 121: -1, 122: -1, 123: -1, 124: 30, 125: 31, 126: -1, 127: -1, 128: -1, 129: -1, 130: 32, 131: -1, 132: 33, 133: -1, 134: -1, 135: -1, 136: -1, 137: -1, 138: -1, 139: -1, 140: -1, 141: -1, 142: -1, 143: 34, 144: 35, 145: -1, 146: -1, 147: -1, 148: -1, 149: -1, 150: 36, 151: 37, 152: -1, 153: -1, 154: -1, 155: -1, 156: -1, 157: -1, 158: -1, 159: -1, 160: -1, 161: -1, 162: -1, 163: -1, 164: -1, 165: -1, 166: -1, 167: -1, 168: -1, 169: -1, 170: -1, 171: -1, 172: -1, 173: -1, 174: -1, 175: -1, 176: -1, 177: -1, 178: -1, 179: -1, 180: -1, 181: -1, 182: -1, 183: -1, 184: -1, 185: -1, 186: -1, 187: -1, 188: -1, 189: -1, 190: -1, 191: -1, 192: -1, 193: -1, 194: -1, 195: -1, 196: -1, 197: -1, 198: -1, 199: -1, 200: -1, 201: -1, 202: -1, 203: -1, 204: -1, 205: -1, 206: -1, 207: 38, 208: -1, 209: -1, 210: -1, 211: -1, 212: -1, 213: -1, 214: -1, 215: -1, 216: -1, 217: -1, 218: -1, 219: -1, 220: -1, 221: -1, 222: -1, 223: -1, 224: -1, 225: -1, 226: -1, 227: -1, 228: -1, 229: -1, 230: -1, 231: -1, 232: -1, 233: -1, 234: 39, 235: 40, 236: -1, 237: -1, 238: -1, 239: -1, 240: -1, 241: -1, 242: -1, 243: -1, 244: -1, 245: -1, 246: -1, 247: -1, 248: -1, 249: -1, 250: -1, 251: -1, 252: -1, 253: -1, 254: 41, 255: -1, 256: -1, 257: -1, 258: -1, 259: -1, 260: -1, 261: -1, 262: -1, 263: -1, 264: -1, 265: -1, 266: -1, 267: -1, 268: -1, 269: -1, 270: -1, 271: -1, 272: -1, 273: -1, 274: -1, 275: -1, 276: -1, 277: 42, 278: -1, 279: -1, 280: -1, 281: -1, 282: -1, 283: 43, 284: -1, 285: -1, 286: -1, 287: 44, 288: -1, 289: -1, 290: -1, 291: 45, 292: -1, 293: -1, 294: -1, 295: 46, 296: -1, 297: -1, 298: 47, 299: -1, 300: -1, 301: 48, 302: -1, 303: -1, 304: -1, 305: -1, 306: 49, 307: 50, 308: 51, 309: 52, 310: 53, 311: 54, 312: -1, 313: 55, 314: 56, 315: 57, 316: -1, 317: 58, 318: -1, 319: 59, 320: -1, 321: -1, 322: -1, 323: 60, 324: 61, 325: -1, 326: 62, 327: 63, 328: -1, 329: -1, 330: 64, 331: -1, 332: -1, 333: -1, 334: 65, 335: 66, 336: 67, 337: -1, 338: -1, 339: -1, 340: -1, 341: -1, 342: -1, 343: -1, 344: -1, 345: -1, 346: -1, 347: 68, 348: -1, 349: -1, 350: -1, 351: -1, 352: -1, 353: -1, 354: -1, 355: -1, 356: -1, 357: -1, 358: -1, 359: -1, 360: -1, 361: 69, 362: -1, 363: 70, 364: -1, 365: -1, 366: -1, 367: -1, 368: -1, 369: -1, 370: -1, 371: -1, 372: 71, 373: -1, 374: -1, 375: -1, 376: -1, 377: -1, 378: 72, 379: -1, 380: -1, 381: -1, 382: -1, 383: -1, 384: -1, 385: -1, 386: 73, 387: -1, 388: -1, 389: -1, 390: -1, 391: -1, 392: -1, 393: -1, 394: -1, 395: -1, 396: -1, 397: 74, 398: -1, 399: -1, 400: 75, 401: 76, 402: 77, 403: -1, 404: 78, 405: -1, 406: -1, 407: 79, 408: -1, 409: -1, 410: -1, 411: 80, 412: -1, 413: -1, 414: -1, 415: -1, 416: 81, 417: 82, 418: -1, 419: -1, 420: 83, 421: -1, 422: -1, 423: -1, 424: -1, 425: 84, 426: -1, 427: -1, 428: 85, 429: -1, 430: 86, 431: -1, 432: -1, 433: -1, 434: -1, 435: -1, 436: -1, 437: 87, 438: 88, 439: -1, 440: -1, 441: -1, 442: -1, 443: -1, 444: -1, 445: 89, 446: -1, 447: -1, 448: -1, 449: -1, 450: -1, 451: -1, 452: -1, 453: -1, 454: -1, 455: -1, 456: 90, 457: 91, 458: -1, 459: -1, 460: -1, 461: 92, 462: 93, 463: -1, 464: -1, 465: -1, 466: -1, 467: -1, 468: -1, 469: -1, 470: 94, 471: -1, 472: 95, 473: -1, 474: -1, 475: -1, 476: -1, 477: -1, 478: -1, 479: -1, 480: -1, 481: -1, 482: -1, 483: 96, 484: -1, 485: -1, 486: 97, 487: -1, 488: 98, 489: -1, 490: -1, 491: -1, 492: 99, 493: -1, 494: -1, 495: -1, 496: 100, 497: -1, 498: -1, 499: -1, 500: -1, 501: -1, 502: -1, 503: -1, 504: -1, 505: -1, 506: -1, 507: -1, 508: -1, 509: -1, 510: -1, 511: -1, 512: -1, 513: -1, 514: 101, 515: -1, 516: 102, 517: -1, 518: -1, 519: -1, 520: -1, 521: -1, 522: -1, 523: -1, 524: -1, 525: -1, 526: -1, 527: -1, 528: 103, 529: -1, 530: 104, 531: -1, 532: -1, 533: -1, 534: -1, 535: -1, 536: -1, 537: -1, 538: -1, 539: 105, 540: -1, 541: -1, 542: 106, 543: 107, 544: -1, 545: -1, 546: -1, 547: -1, 548: -1, 549: 108, 550: -1, 551: -1, 552: 109, 553: -1, 554: -1, 555: -1, 556: -1, 557: 110, 558: -1, 559: -1, 560: -1, 561: 111, 562: 112, 563: -1, 564: -1, 565: -1, 566: -1, 567: -1, 568: -1, 569: 113, 570: -1, 571: -1, 572: 114, 573: 115, 574: -1, 575: 116, 576: -1, 577: -1, 578: -1, 579: 117, 580: -1, 581: -1, 582: -1, 583: -1, 584: -1, 585: -1, 586: -1, 587: -1, 588: -1, 589: 118, 590: -1, 591: -1, 592: -1, 593: -1, 594: -1, 595: -1, 596: -1, 597: -1, 598: -1, 599: -1, 600: -1, 601: -1, 602: -1, 603: -1, 604: -1, 605: -1, 606: 119, 607: 120, 608: -1, 609: 121, 610: -1, 611: -1, 612: -1, 613: -1, 614: 122, 615: -1, 616: -1, 617: -1, 618: -1, 619: -1, 620: -1, 621: -1, 622: -1, 623: -1, 624: -1, 625: -1, 626: 123, 627: 124, 628: -1, 629: -1, 630: -1, 631: -1, 632: -1, 633: -1, 634: -1, 635: -1, 636: -1, 637: -1, 638: -1, 639: -1, 640: 125, 641: 126, 642: 127, 643: 128, 644: -1, 645: -1, 646: -1, 647: -1, 648: -1, 649: -1, 650: -1, 651: -1, 652: -1, 653: -1, 654: -1, 655: -1, 656: -1, 657: -1, 658: 129, 659: -1, 660: -1, 661: -1, 662: -1, 663: -1, 664: -1, 665: -1, 666: -1, 667: -1, 668: 130, 669: -1, 670: -1, 671: -1, 672: -1, 673: -1, 674: -1, 675: -1, 676: -1, 677: 131, 678: -1, 679: -1, 680: -1, 681: -1, 682: 132, 683: -1, 684: 133, 685: -1, 686: -1, 687: 134, 688: -1, 689: -1, 690: -1, 691: -1, 692: -1, 693: -1, 694: -1, 695: -1, 696: -1, 697: -1, 698: -1, 699: -1, 700: -1, 701: 135, 702: -1, 703: -1, 704: 136, 705: -1, 706: -1, 707: -1, 708: -1, 709: -1, 710: -1, 711: -1, 712: -1, 713: -1, 714: -1, 715: -1, 716: -1, 717: -1, 718: -1, 719: 137, 720: -1, 721: -1, 722: -1, 723: -1, 724: -1, 725: -1, 726: -1, 727: -1, 728: -1, 729: -1, 730: -1, 731: -1, 732: -1, 733: -1, 734: -1, 735: -1, 736: 138, 737: -1, 738: -1, 739: -1, 740: -1, 741: -1, 742: -1, 743: -1, 744: -1, 745: -1, 746: 139, 747: -1, 748: -1, 749: 140, 750: -1, 751: -1, 752: 141, 753: -1, 754: -1, 755: -1, 756: -1, 757: -1, 758: 142, 759: -1, 760: -1, 761: -1, 762: -1, 763: 143, 764: -1, 765: 144, 766: -1, 767: -1, 768: 145, 769: -1, 770: -1, 771: -1, 772: -1, 773: 146, 774: 147, 775: -1, 776: 148, 777: -1, 778: -1, 779: 149, 780: 150, 781: -1, 782: -1, 783: -1, 784: -1, 785: -1, 786: 151, 787: -1, 788: -1, 789: -1, 790: -1, 791: -1, 792: 152, 793: -1, 794: -1, 795: -1, 796: -1, 797: 153, 798: -1, 799: -1, 800: -1, 801: -1, 802: 154, 803: 155, 804: 156, 805: -1, 806: -1, 807: -1, 808: -1, 809: -1, 810: -1, 811: -1, 812: -1, 813: 157, 814: -1, 815: 158, 816: -1, 817: -1, 818: -1, 819: -1, 820: 159, 821: -1, 822: -1, 823: 160, 824: -1, 825: -1, 826: -1, 827: -1, 828: -1, 829: -1, 830: -1, 831: 161, 832: -1, 833: 162, 834: -1, 835: 163, 836: -1, 837: -1, 838: -1, 839: 164, 840: -1, 841: -1, 842: -1, 843: -1, 844: -1, 845: 165, 846: -1, 847: 166, 848: -1, 849: -1, 850: 167, 851: -1, 852: -1, 853: -1, 854: -1, 855: -1, 856: -1, 857: -1, 858: -1, 859: 168, 860: -1, 861: -1, 862: 169, 863: -1, 864: -1, 865: -1, 866: -1, 867: -1, 868: -1, 869: -1, 870: 170, 871: -1, 872: -1, 873: -1, 874: -1, 875: -1, 876: -1, 877: -1, 878: -1, 879: 171, 880: 172, 881: -1, 882: -1, 883: -1, 884: -1, 885: -1, 886: -1, 887: -1, 888: 173, 889: -1, 890: 174, 891: -1, 892: -1, 893: -1, 894: -1, 895: -1, 896: -1, 897: 175, 898: -1, 899: -1, 900: 176, 901: -1, 902: -1, 903: -1, 904: -1, 905: -1, 906: -1, 907: 177, 908: -1, 909: -1, 910: -1, 911: -1, 912: -1, 913: 178, 914: -1, 915: -1, 916: -1, 917: -1, 918: -1, 919: -1, 920: -1, 921: -1, 922: -1, 923: -1, 924: 179, 925: -1, 926: -1, 927: -1, 928: -1, 929: -1, 930: -1, 931: -1, 932: 180, 933: 181, 934: 182, 935: -1, 936: -1, 937: 183, 938: -1, 939: -1, 940: -1, 941: -1, 942: -1, 943: 184, 944: -1, 945: 185, 946: -1, 947: 186, 948: -1, 949: -1, 950: -1, 951: 187, 952: -1, 953: -1, 954: 188, 955: -1, 956: 189, 957: 190, 958: -1, 959: 191, 960: -1, 961: -1, 962: -1, 963: -1, 964: -1, 965: -1, 966: -1, 967: -1, 968: -1, 969: -1, 970: -1, 971: 192, 972: 193, 973: -1, 974: -1, 975: -1, 976: -1, 977: -1, 978: -1, 979: -1, 980: 194, 981: 195, 982: -1, 983: -1, 984: 196, 985: -1, 986: 197, 987: 198, 988: 199, 989: -1, 990: -1, 991: -1, 992: -1, 993: -1, 994: -1, 995: -1, 996: -1, 997: -1, 998: -1, 999: -1}
indices_in_1k = [k for k in thousand_k_to_200 if thousand_k_to_200[k] != -1]

# compute the output distribution
def get_predictions(images, model, transforms, device):
  # collect the prediction for every image in input
  img_results = []
  for img in images:
    single_batch = transforms(img).unsqueeze(0).to(device)
    prediction = model(single_batch).squeeze(0)
    img_results.append(prediction[indices_in_1k])
  return torch.stack(img_results).to(device)

# compute the marginal cross entropy
def marginal_cross_entropy(marginal_dist, labels, cost_function):
  entropy = 0.0
  # sum all entropies for the different labels since I don't know the real one
  for label in labels:
    entropy += cost_function(marginal_dist, label)
  return entropy

def marginal_entropy_loss(logits):
    """
    Computes the entropy of the marginal output distribution over multiple augmentations.

    Args:
        logits (torch.Tensor): A tensor of shape (N, C), where N is the number of augmentations
                               and C is the number of classes. Each row corresponds to the logits
                               produced by the model for one augmentation.

    Returns:
        torch.Tensor: The scalar value of the marginal entropy loss.
    """
    # Step 1: Normalize logits to get log probabilities
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)

    # Step 2: Compute the average log probabilities over the augmentations
    avg_log_probs = log_probs.logsumexp(dim=0) - torch.log(torch.tensor(logits.shape[0], dtype=log_probs.dtype))

    # Step 3: Calculate the marginal probability distribution (exp of avg_log_probs)
    avg_probs = torch.exp(avg_log_probs)

    # Step 4: Compute the entropy of the marginal distribution
    entropy_loss = -(avg_probs * avg_log_probs).sum()

    return entropy_loss


# test time robustness via MEMO algorithm
def adapt(model, test_sample, B, optimizer, transforms, device):
  
  with torch.enable_grad():
    # get the B + 1 images
    augmented_images = augment_image(test_sample, augmentations, B)

    # get results for each image
    raw_logits = get_predictions(augmented_images, model, transforms, device)

    # compute marginal distribution on raw logits
    loss = marginal_entropy_loss(raw_logits)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test(model, data_loader, B, cost_function, optimizer, transforms, device="cuda"):
  correct = []

  # save the original model weights
  original_params = copy.deepcopy(model.state_dict())

  # set the network to evaluation mode
  model.eval()

  # iterate over the test set
  for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader), desc="Testing", leave=False):
    # Load data into GPU
    inputs = inputs.to(device)
    targets = targets.to(device)

    # apply MEMO to each test point in the batch
    intermediate_outputs = []
    for input, label in zip(inputs, targets):
      # adapt weights by performing marginal output minimization
      adapt(model, input, B, optimizer, transforms, device)
      # intermediate_outputs.append(output)

      with torch.no_grad():
        outputs = model(input.unsqueeze(0))[:, indices_in_1k]

        _, predicted = outputs.max(1)
        confidence = torch.nn.functional.softmax(outputs, dim=1).squeeze()[predicted].item() 
      correctness = 1 if predicted.item() == label.item() else 0
      correct.append(correctness)

      # reapply original weights to the model
      model.load_state_dict(original_params)

    if (batch_idx+1) % 10 == 0:
      print(f'MEMO adapt test error {(1-np.mean(correct))*100:.2f}')
  print(f'Final MEMO adapt test error {(1-np.mean(correct))*100:.2f}')


import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Stage 1: Augmentation Prediction Network
class AugmentationNetwork(nn.Module):
    def __init__(self, num_augmentations=10):
        super(AugmentationNetwork, self).__init__()
        weights = ResNet50_Weights.DEFAULT
        self.resnet50 = models.resnet50(weights)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_augmentations)
    
    def forward(self, x):
        # Output the probabilities of each augmentation
        aug_params = self.resnet50(x)
        return aug_params

# Stage 2: Classification Network
class ClassificationNetwork(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationNetwork, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet50 = models.resnet18(weights)
        # self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)
    
    def forward(self, x):
        # Output the class probabilities
        class_probs = self.resnet50(x)
        return class_probs

# Main Two-Stage Network
class TwoStageNetwork(nn.Module):
    def __init__(self, num_augmentations=4, num_classes=1000):
        self.num_aug = num_augmentations
        super(TwoStageNetwork, self).__init__()
        self.augmentation_net = AugmentationNetwork(num_augmentations)
        self.classification_net = ClassificationNetwork(num_classes)
        
        # Define possible augmentations
        self.augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.5),
            # transforms.ColorJitter(contrast=0.5),
            # transforms.ColorJitter(saturation=0.5),
            # transforms.ColorJitter(hue=0.5),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
            # transforms.RandomResizedCrop(224)
        ]
        
    def forward(self, x):
      # Generate multiple augmented versions of each image
      # aug_indices = torch.arange(self.num_aug).unsqueeze(0).repeat(x.shape[0], 1)  # All augmentations
      augmented_images = torch.stack([aug(x) for aug in self.augmentations], dim=1)  
      
      # Stage 1: Predict augmentation
      aug_pred = self.augmentation_net(augmented_images.view(-1, 3, 224, 224))
      aug_index = torch.argmax(aug_pred, dim=1)
      
      # Apply selected augmentation
      aug_x = torch.stack([self.augmentations[i](img) for img, i in zip(x, aug_index)])

      # Forward pass through classification network for each augmentation
      cls_preds = self.classification_net(aug_x)
      
      # Stage 2: Classify augmented image
      # class_probs = self.classification_net(aug_x)
      return cls_preds, aug_index



def train(
    run_name,
    num_epochs,
    batch_size = 4,
    device = "cuda",
    learning_rate=0.00025,
    weight_decay=0.000001,
    momentum=0.9,
    num_augmentations = 4
):
   
  # itialize the ResNet model
  weights = ResNet50_Weights.DEFAULT
  # initialize the inference transforms
  preprocess = weights.transforms()
  # initialize the test dataloader
  test_loader, _ = get_data(batch_size, data_folder, preprocess)

  # Instantiate the model
  model = TwoStageNetwork(num_augmentations=4, num_classes=200)
  model = model.to(device)
  # Freeze the classification network's weights
  # for param in model.classification_net.parameters():
  #     param.requires_grad = False

  # Verify that the augmentation network parameters require grad
  # for param in model.augmentation_net.parameters():
  #     param.requires_grad = True

  criterion = torch.nn.CrossEntropyLoss()
  # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  # Only optimize the augmentation network's parameters
  optimizer = torch.optim.Adam(model.augmentation_net.parameters(), lr=0.0001)
  total=0.0
  correct=0
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      
      class_probs, aug_index = model(images)
      class_probs = class_probs[:, indices_in_1k]
      # Compute classification loss for each augmentation
      loss_classification = criterion(class_probs, labels)

      # Backpropagation and optimization
      loss_classification.backward()
      optimizer.step()
      
      # Track statistics
      running_loss += loss_classification.item()
      _, predicted = torch.max(class_probs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      
    # Print statistics every few batches
    # if (epoch + 1) % 2 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{epoch+1}/{len(test_loader)}], '
          f'Loss: {running_loss / (epoch+1):.4f}, Accuracy: {100 * correct / total:.2f}%')
    

      # Calculate total loss and gradients
      # total_loss = torch.mean(torch.stack([cls_losses[i, aug_index[i]] for i in range(images.shape[0])]))
      # print(cls_losses)

    #   Forward pass
    #   aug_pred, class_pred = model(images)
    
    #   # Compute the classification loss only
    #   loss = criterion(class_pred, class_labels)
      
    #   # Backward pass and optimization
    #   loss.backward()
    #   optimizer.step()
      
    #   running_loss += loss.item()
  
    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(test_loader)}")

def main_test(
    run_name,
    batch_size = 32,
    device = "cuda",
    learning_rate=0.00025,
    weight_decay=0.000001,
    momentum=0.9,
    num_augmentations = 31
):
  # writer = SummaryWriter(log_dir=f"runs/{run_name}")
  device = device

  # itialize the ResNet model
  weights = ResNet50_Weights.DEFAULT
  model = resnet50(weights=weights).to(device)

  # initialize the inference transforms
  preprocess = weights.transforms()
  preprocess

  # initialize the test dataloader
  test_loader, _ = get_data(batch_size, data_folder, preprocess)

  # initialize the optimizer
  optimizer = get_optimizer(model, learning_rate, weight_decay, momentum)

  # initialize the cost function
  cost_function = get_cost_function()

  test(model, test_loader, num_augmentations, cost_function, optimizer, preprocess, device)
  # print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")


train("resnet_memo", 50)
# main_test("resnet_MEMO")
