import ismrmrd
import os
import itertools
import logging
import traceback
import numpy as np
import numpy.fft as fft
import xml.dom.minidom
import base64
import ctypes
import re
import mrdhelper
import constants
from time import perf_counter

### Imported to generate LVEF figure ###
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

### IMPORTED FOR SEGMENTATION (START) ###
import os
import glob
import numpy as np

# modules for extracting the contours
from skimage import measure
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_fill_holes

import sys
sys.path.append("/mnt/alp/Users/Manuel/code/DeepStrain/")

# modules for generating the segmentation
from models import deep_strain_model
from tensorflow.keras.optimizers import Adam
from data import base_dataset
### IMPORTED FOR SEGMENTATION (END) ###

# Folder for debug output files
debugFolder = "/tmp/share/debug"

class Options():
    def __init__(self):
        self.datadir = '../../../../datasets/ACDC'
        self.isTrain = False
        self.image_shape = (128,128,1)
        self.volume_shape = (128,128,16,1)
        self.nlabels = 4
        self.pretrained_models_netS  = '/home/mmorales/main_python/DeepStrain/pretrained_models/carson_Jan2021.h5'
        self.pretrained_models_netME = '/home/mmorales/main_python/DeepStrain/pretrained_models/carmen_Jan2021.h5'

def get_segmentation_model():
    opt = Options()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # use cpu
    model = deep_strain_model.DeepStrain(Adam, opt)
    netS  = model.get_netS()
    return netS

netS = get_segmentation_model()

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)    
    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))

        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3 and a matrix size of (%s x %s x %s)", 
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    currentSeries = 0
    imgGroup = []
    try:
        for item in connection:
          # ----------------------------------------------------------
          # Image data messages
          # ----------------------------------------------------------
          if isinstance(item, ismrmrd.Image):
              # When this criteria is met, run process_group() on the accumulated
              # data, which returns images that are sent back to the client.
              # e.g. when the series number changes:
              if item.image_series_index != currentSeries:
                  logging.info("Processing a group of images because series index changed to %d", item.image_series_index)
                  currentSeries = item.image_series_index
                  image = process_image(imgGroup, connection, config, metadata)
                  connection.send_image(image)
                  imgGroup = []

              # Only process magnitude images -- send phase images back without modification (fallback for images with unknown type)
              if (item.image_type is ismrmrd.IMTYPE_MAGNITUDE) or (item.image_type == 0):
                  imgGroup.append(item)
              else:
                  tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                  tmpMeta['Keep_image_geometry']    = 1
                  item.attribute_string = tmpMeta.serialize()

                  connection.send_image(item)
                  continue

          elif item is None:
              break

          else:
              logging.error("Unsupported data type %s", type(item).__name__)

        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, connection, config, metadata)
            connection.send_image(image)
            imgGroup = []

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())

    finally:
        connection.send_close()

def process_image(images, connection, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # Note: The MRD Image class stores data as [cha z y x]

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data                              for img in images])
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    # Reformat data to [y x z cha img], i.e. [row col] for the first two dimensions
    data = data.transpose((3, 4, 2, 1, 0))

    # Display MetaAttributes for first image
    logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(meta[0]))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in meta[0]:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(meta[0]['IceMiniHead']).decode('utf-8'))

    logging.debug("Original image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgOrig.npy", data)

    # Normalize and convert to int16
    data = data.astype(np.float64)
    data *= 32767/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    currentSeries = 1

    total_phase_count, total_slice_count = head[-1].phase+1, head[-1].slice+1

    blood_pool_buff = np.array([[None] * total_phase_count] * total_slice_count)
    myocardium_buff = np.array([[None] * total_phase_count] * total_slice_count)

    print ('DATA SHAPE:', data.shape)

    # Re-slice back into 2D images
    imagesOut = [None] * data.shape[-1]

    for iImg in range(data.shape[-1]):
        # Create new MRD instance for the inverted image
        # Transpose from convenience shape of [y x z cha] to MRD Image shape of [cha z y x]
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        imagesOut[iImg] = ismrmrd.Image.from_array(data[...,iImg].transpose((3, 2, 0, 1)), transpose=False)
        data_type = imagesOut[iImg].data_type

        # Extract slice and voxel thickness information
        slice_thickness = imagesOut[iImg].field_of_view[2]
        pixel_spacing_x = float(imagesOut[iImg].field_of_view[0]) / imagesOut[iImg].data.shape[2]
        pixel_spacing_y = float(imagesOut[iImg].field_of_view[1]) / imagesOut[iImg].data.shape[3]

        # Create a copy of the original fixed header and update the data_type
        # (we changed it to int16 from all other types)
        oldHeader = head[iImg]
        oldHeader.data_type = data_type
        oldHeader.image_index = head[iImg].phase + head[iImg].slice*total_phase_count + 1

        # Unused example, as images are grouped by series before being passed into this function now
        # oldHeader.image_series_index = currentSeries

        # Increment series number when flag detected (i.e. follow ICE logic for splitting series)
        if mrdhelper.get_meta_value(meta[iImg], 'IceMiniHead') is not None:
            if mrdhelper.extract_minihead_bool_param(base64.b64decode(meta[iImg]['IceMiniHead']).decode('utf-8'), 'BIsSeriesEnd') is True:
                currentSeries += 1

        imagesOut[iImg].setHead(oldHeader)

        minval=np.min(data[:,:,0,0,iImg])
        maxval=np.max(data[:,:,0,0,iImg])

        center = (minval+maxval)/2
        window = int(center*2)

        # Create a copy of the original ISMRMRD Meta attributes and update
        tmpMeta                                   = meta[iImg]
        tmpMeta['DataRole']                       = 'Image'
        tmpMeta['ImageProcessingHistory']         = ['PYTHON', 'INVERT']
        tmpMeta['WindowCenter']                   = str(center)
        tmpMeta['WindowWidth']                    = str(window)
        tmpMeta['SequenceDescriptionAdditional']  = 'FIRE'
        tmpMeta['Keep_image_geometry']            = 1

        # Example for setting colormap
        # tmpMeta['LUTFileName']            = 'MicroDeltaHotMetal.pal'

        # Add image orientation directions to MetaAttributes if not already present
        if tmpMeta.get('ImageRowDir') is None:
            tmpMeta['ImageRowDir'] = ["{:.18f}".format(oldHeader.read_dir[0]), "{:.18f}".format(oldHeader.read_dir[1]), "{:.18f}".format(oldHeader.read_dir[2])]

        if tmpMeta.get('ImageColumnDir') is None:
            tmpMeta['ImageColumnDir'] = ["{:.18f}".format(oldHeader.phase_dir[0]), "{:.18f}".format(oldHeader.phase_dir[1]), "{:.18f}".format(oldHeader.phase_dir[2])]

        roi_list, image_mask = create_roi_from_contour(data[:,:,0,0,iImg])

        for index, roi in enumerate(roi_list):
            roi_tag = 'ROI_' + str(iImg) + str(index)
            tmpMeta[roi_tag] = roi

        blood_pool_voxels_count = np.squeeze(image_mask==3).sum(axis=(0,1))
        myocardium_voxels_count = np.squeeze(image_mask==2).sum(axis=(0,1))

        blood_pool_buff[head[iImg].slice, head[iImg].phase] = blood_pool_voxels_count
        myocardium_buff[head[iImg].slice, head[iImg].phase] = myocardium_voxels_count

        metaXml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
        logging.debug("Image data has %d elements", imagesOut[iImg].data.size)

        imagesOut[iImg].attribute_string = metaXml

    pixel_spacing_x = float(head[0].field_of_view[0]) / head[0].matrix_size[0]
    pixel_spacing_y = float(head[0].field_of_view[1]) / head[0].matrix_size[1]
    pixel_spacing_z = head[0].field_of_view[2]

    # Compute Slice Gap
    first_image = imagesOut[0]
    last_image = imagesOut[-1]
    ImagePositionPatient_0 = np.array([first_image.position[0], first_image.position[1], first_image.position[2]])
    ImagePositionPatient_1 = np.array([last_image.position[0], last_image.position[1], last_image.position[2]])
    distance_between_planes = np.linalg.norm(ImagePositionPatient_0-ImagePositionPatient_1) / imagesOut[-1].slice # (First Slice Position - Final Slice Position) / Slice Count
    slice_gap = distance_between_planes - pixel_spacing_z
    slice_thickness = pixel_spacing_z + slice_gap/2

    print('SLICE THICKNESS:', slice_thickness)

    voxel_volume = (pixel_spacing_x * pixel_spacing_y * slice_thickness)/1000

    lvef_figure = create_lvef_figure(blood_pool_buff, myocardium_buff, voxel_volume)

    # Update fixed header so it has field of view information
    lvef_figure_head = lvef_figure.getHead()
    lvef_figure_head.field_of_view = oldHeader.field_of_view
    lvef_figure_head.image_series_index = (oldHeader.image_series_index + 1)
    # lvef_figure_head.image_type = 6  # To be defined as ismrmrd.IMTYPE_RGB
    # lvef_figure_head.channels   = 3  # RGB "channels".  This is set by from_array, but need to be explicit as we're copying the old header instead
    lvef_figure.setHead(lvef_figure_head)

    # Update flexiable header so it has phase direction information
    lvef_figure_meta = ismrmrd.Meta.deserialize(lvef_figure.attribute_string)
    lvef_figure_meta['Keep_image_geometry'] = 1
    
    # lvef_figure_meta['ImageRowDir'] = ["{:.18f}".format(oldHeader.read_dir[0]), "{:.18f}".format(oldHeader.read_dir[1]), "{:.18f}".format(oldHeader.read_dir[2])]
    # lvef_figure_meta['ImageColumnDir'] = ["{:.18f}".format(oldHeader.phase_dir[0]), "{:.18f}".format(oldHeader.phase_dir[1]), "{:.18f}".format(oldHeader.phase_dir[2])]
    # lvef_figure_meta['ImagesliceNormDir'] = ["{:.18f}".format(oldHeader.slice_dir[0]), "{:.18f}".format(oldHeader.slice_dir[1]), "{:.18f}".format(oldHeader.slice_dir[2])]
    
    lvef_figure_meta['ImageRowDir'] = mrdhelper.get_meta_value(ismrmrd.Meta.deserialize(images[-1].attribute_string), 'ImageRowDir')
    lvef_figure_meta['ImageColumnDir'] = mrdhelper.get_meta_value(ismrmrd.Meta.deserialize(images[-1].attribute_string), 'ImageColumnDir')
    lvef_figure_meta['ImagesliceNormDir'] = mrdhelper.get_meta_value(ismrmrd.Meta.deserialize(images[-1].attribute_string), 'ImageSliceNormDir')
    
    metaXml = lvef_figure_meta.serialize()
    lvef_figure.attribute_string = metaXml

    # Append figure to output array
    imagesOut.append(lvef_figure)

    return imagesOut

def create_roi_from_contour(img):
    img = img[..., np.newaxis, np.newaxis]
    image_mask = get_mask(img, netS)
    # image_mask = img > np.percentile(img.ravel(), 80)

    contours = mask_to_contours(image_mask)
    roi_list = []
    roi_colors = [(1,1,0),(0,1,0),(1,0,0)] # RV = Yellow, LV Epi = Green, LV Endo = Red

    for i, c in enumerate(contours):
        x = c[:, 1]
        y = c[:, 0]

        rgb = roi_colors[i % len(roi_colors)] # Red, green, blue color -- normalized to 1
        thickness  = 1 # Line thickness
        style      = 0 # Line style (0 = solid, 1 = dashed)
        visibility = 1 # Line visibility (0 = false, 1 = true)

        roi = mrdhelper.create_roi(x, y, rgb, thickness, style, visibility)

        roi_list.append(roi)
    
    return roi_list, image_mask

def Contours(mask, tissue_labels=[1,2,3]):
    contours = []
    for i in tissue_labels:
        mask_ = binary_fill_holes(mask==i)
        c = measure.find_contours(mask_,0.8)
        if len(c) == 0:  continue
        c = c[np.argmax([len(c) for c in c])]
        contours.append(c)
    return contours

def normalize(x, axis=(0,1,2)):
    # normalize per volume (x,y,z) frame
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x-mu)/(sd+1e-8)

def get_mask(Array4D, netS):
    """Segmentation of LV and RV blood pools and LV myocardium. 
    Input
    -----
    Array4D : Array of shape n_ro, n_pe, n_slices, n_frames
    """
    n_ro, n_pe, n_slices, n_frames = Array4D.shape
    print('INPUT TO NETWORK:', Array4D.shape)
    Array4D_ROI = np.zeros((n_ro, n_pe, n_slices, n_frames), dtype=np.int16)
    Array4D = Array4D.transpose((2,3,0,1)).reshape((-1,n_ro,n_pe)) # (n_slices*n_frames,n_ro,ny)
    Array4D = normalize(Array4D, axis=(1,2))
    M = netS(Array4D[:,n_ro//2-64:n_ro//2+64,n_pe//2-64:n_pe//2+64,None])
    Array4D_ROI[n_ro//2-64:n_ro//2+64,n_pe//2-64:n_pe//2+64] += np.argmax(M, -1).transpose((1,2,0)).reshape((128,128,n_slices,n_frames))
    return Array4D_ROI

def mask_to_contours(mask, tissue_labels=[1,2,3]):
    """ Extract contours of 2D mask 
    Returns
    -------
    Contours : list of contours, where each contour is of shape (n_points, 2) for x, y coordinates.
    """
    Contours = []
    mask = np.squeeze(mask)
    print('SHAPE OF MASK', mask.shape)
    print('MIN AND MAX OF MASK', mask.min(), mask.max())
    for i in tissue_labels:
        mask_ = binary_fill_holes(mask==i)
        c = measure.find_contours(mask_,0.8)
        if len(c) == 0:  continue
        c = c[np.argmax([len(c) for c in c])]
        Contours.append(c)
    
    return Contours

def create_lvef_figure(blood_pool_buff, myocardium_buff, voxel_volume):

    edv_blood_pool_total = 0
    esv_blood_pool_total = 0

    edv_index = np.argmax(blood_pool_buff.sum(axis=0))
    esv_index = np.argmin(blood_pool_buff.sum(axis=0))

    for z_slice in range(blood_pool_buff.shape[0]):

      ef_slice = blood_pool_buff[z_slice]

      edv_blood_pool_voxels = ef_slice[edv_index]
      esv_blood_pool_voxels = ef_slice[esv_index]

      edv_blood_pool_total += (edv_blood_pool_voxels*voxel_volume)
      esv_blood_pool_total += (esv_blood_pool_voxels*voxel_volume)

    strove_volume = np.round((edv_blood_pool_total-esv_blood_pool_total), 2)
    ejection_fraction = np.round((strove_volume / edv_blood_pool_total)*100, 2)
    edv_blood_pool_total = np.round(edv_blood_pool_total, 2)
    esv_blood_pool_total = np.round(esv_blood_pool_total, 2)

    # print('EF:', ejection_fraction, 'SV:', strove_volume, 'EDV:', edv_blood_pool_total, 'ESV:', esv_blood_pool_total)

    # Generate a figure and attach it to a canvas.
    fig = Figure(figsize=(5, 4), dpi=100)
    # Attach it to a canvas
    canvas = FigureCanvasAgg(fig)

    ax = fig.add_subplot(111)

    table_data = [
    ['EDV (Phase ' + str(edv_index + 1) + ')', str(edv_blood_pool_total) + ' ml'],
    ['ESV (Phase ' + str(esv_index + 1) + ')', str(esv_blood_pool_total) + ' ml'],
    ['SV', str(strove_volume) + ' ml'],
    ['EF', str(ejection_fraction) + ' %']]

    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')

    # Modify table
    table.set_fontsize(14)
    table.scale(1,4)
    ax.axis('off')

    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()

    # convert to a NumPy array
    X = np.asarray(buf)
    X = X.astype(np.int16)

    # Create MRD image
    lvef_figure = ismrmrd.Image.from_array(X[:,:,0], transpose=False)

    return lvef_figure
