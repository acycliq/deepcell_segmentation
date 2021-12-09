from deepcell.applications import CytoplasmSegmentation, Mesmer
import numpy as np
import pandas as pd
import diplib as dip
import skimage.io
from PIL import Image, ImageDraw
import utils
import os
import logging

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

# ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = r"D:\Home\Dimitris\OneDrive - University College London\Data\Izzie\210514 ATPase + Cadherin antibody test - 1-100 -secondary\downscaled"
FRAMES_JPG_DIR = os.path.join(ROOT_DIR, 'anti cadherin', 'frames', 'jpg')
BOUNDARIES_JPG_DIR = os.path.join(ROOT_DIR, 'anti cadherin', 'out', 'boundaries')


def extract_borders_dip(label_image, offset_x=0, offset_y=0, ignore_labels=[0]):
    """
    Takes in a label_image and extracts the boundaries. It is assumed that the background
    has label = 0
    Parameters
    ----------
    label_image: the segmentation mask, a 2-D array
    offset_x: Determines how much the boundaries will be shifted by the on the x-axis
    offset_y: Determines how much the boundaries will be shifted by the on the y-axis
    ignore_labels: list of integers. If you want to ignore some masks, put the corresponding label in this list

    Returns
    -------
    Returns a dataframa with two columns, The first one is the mask label and the second column keeps the coordinates
    of the mask boundaries
    """
    labels = sorted(set(label_image.flatten()) - {0} - set(ignore_labels))
    cc = dip.GetImageChainCodes(label_image)  # input must be an unsigned integer type
    d = {}
    for c in cc:
        if c.objectID in labels:
            # p = np.array(c.Polygon())
            p = c.Polygon().Simplify()
            p = p + np.array([offset_x, offset_y])
            p = np.uint64(p).tolist()
            p.append(p[0])  # append the first pair at the end to close the polygon
            d[np.uint64(c.objectID)] = p
        else:
            pass
    df = pd.DataFrame([d]).T
    df = df.reset_index()
    df.columns = ['label', 'coords']
    return df


def get_jpg(i):
    """
    retrieves the jpg to draw the boundaries on
    """
    # 1. first check if there is already a jpg
    jpg_page = os.path.join(BOUNDARIES_JPG_DIR, 'anti cadherin_z%s_c001.jpg' % str(i).zfill(3))
    if os.path.isfile(jpg_page):
        return jpg_page
    else:
        return os.path.join(FRAMES_JPG_DIR, 'anti cadherin_z%s_c001.jpg' % str(i).zfill(3))


def normalize_q(img, q=99):
    X = img.copy()
    X = (X - np.percentile(X, 100-q)) / (np.percentile(X, q) - np.percentile(X, 100-q))
    return X


def convert_to_rgb(page):
    img = normalize_q(page, q=99)
    img = np.clip(img, 0, 1)
    img = img * 255.0
    img = np.stack([img, img, img])
    img = img.astype(np.uint8)
    img = img.transpose((1, 2, 0))
    return img


def draw_poly(polys, colours, jpg_filename):
    img = Image.open(jpg_filename).convert('RGB')

    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    for i, poly in enumerate(polys):
        poly = [tuple(d) for d in poly]
        draw.line(poly, fill=colours[i], width=1)
        # img3 = Image.blend(img, img2, 0.4)
    return img2


def draw_boundaries(img, masks):
    for i, mask in enumerate(masks):
        offset_x = 0
        offset_y = 0
        boundaries = extract_borders_dip(mask.astype(np.uint32), offset_x, offset_y, [0])
        boundaries['colour'] = utils.get_colour(boundaries.label.values)
        polys = boundaries.coords.values
        jpg_filename = get_jpg(i+1)
        if len(polys) > 0:
            res = draw_poly(polys, boundaries['colour'].values, jpg_filename)
            jpg_page = os.path.join(BOUNDARIES_JPG_DIR, os.path.basename(jpg_filename))
            res.save(jpg_page)
            logger.info('Boundaries saved at %s' % jpg_page)

def segment(img):
    app = CytoplasmSegmentation()
    print('Training Resolution:', app.model_mpp, 'microns per pixel')
    y_pred = app.predict(img, image_mpp=1.0, batch_size=2)
    masks_zxy = y_pred[:, :, :, 0]
    masks_stitched = utils.stitch3D(masks_zxy, stitch_threshold=0.5)
    np.savez('deepcell_masks_stitched.npz', masks_stitched)
    logger.info('Masks saved to disk!')
    return masks_zxy


def main(img_path):
    img = skimage.io.imread(img_path)
    img = img[:, :1, :, :]
    img = img.transpose(0, 2, 3, 1)

    masks = segment(img)
    draw_boundaries(img, masks)


if __name__ == "__main__":
    img_path = os.path.join(ROOT_DIR, 'anti cadherin', 'anti cadherin.tif')
    main(img_path)
    logger.info('Done')