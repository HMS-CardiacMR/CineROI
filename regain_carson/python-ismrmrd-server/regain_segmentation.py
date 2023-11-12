import sys
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import xml.dom.minidom
import base64
import ctypes
import re
import mrdhelper
import constants
from time import perf_counter

import torchvision.utils
from PIL import Image

import matplotlib.pyplot as plt

import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Tensor

import neuralnet.Models as Models

### Imported to generate LVEF figure ###
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

### IMPORTED FOR SEGMENTATION (START) ###
import glob

# modules for extracting the contours
from skimage import measure
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_fill_holes

import segmentation_model
### IMPORTED FOR SEGMENTATION (END) ###

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def get_segmentation_model():
    print('Loading segmentation model...')
    netS = segmentation_model.CarSON()
    netS.model.load_weights(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'carson_Jan2021.h5'))
    netS = netS.model
    return netS

netS = get_segmentation_model()

def PercentileRescaler(Arr):
    minval=np.percentile(Arr, 0, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
    maxval=np.percentile(Arr, 100, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)

    if minval==maxval:
        print("Zero Detected")
    Arr=(Arr-minval)/(maxval-minval)
    Arr=np.clip(Arr, 0.0, 1.0)
    return Arr, minval, maxval

def RestoreRescaler (Arr, minval, maxval):
    arr= Arr*(maxval-minval)+(minval)
    arr = np.clip(arr, minval, maxval)
    return arr

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
    acqGroup = []
    imgGroup = []
    waveformGroup = []

    sid=0
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
                    logging.info("imgGroupSize %d",len(imgGroup))
                else:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry']    = 1
                    item.attribute_string = tmpMeta.serialize()

                    connection.send_image(item)
                    continue

        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, connection, config, metadata)
            connection.send_image(image)
            imgGroup = []

    finally:
        connection.send_close()

def image2tensor(image) -> Tensor:
    tensor = torch.from_numpy(np.array(image, np.float32, copy=False))
    return tensor

device = torch.device("cuda:0")
PATH_net = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'neuralnet', 'G_New_epoch75.pth')
generator = Models.GeneratorRRDB(1,filters=64, num_res_blocks=23).to(device)
generator.load_state_dict(torch.load(PATH_net, map_location=device))
torch.no_grad()
generator.eval()
generator.half()

PATH_net2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'neuralnet', 'G_LargeSmall_epoch16.pth')
generator2 = Models.GeneratorRRDB(1,filters=32, num_res_blocks=10).to(device)
generator2.load_state_dict(torch.load(PATH_net2, map_location=device))
torch.no_grad()
generator2.eval()
generator2.half()

def process_image(images, connection, config, metadata):

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data                              for img in images])
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    # Reformat data to the more intuitive [x y z cha img]
    data = data.transpose()

    data = data.transpose((1, 0, 2, 3, 4))

    shapeData = data.shape
    total = shapeData[4] * shapeData[3] * shapeData[2]
    idxim = 0

    for Dim5 in range(0, shapeData[4]):
        for Dim4 in range(0, shapeData[3]):
            for Dim3 in range(0, shapeData[2]):

                lr_image = data[:, :, Dim3, Dim4, Dim5]

                rowdir = meta[idxim].get('ImageRowDir')
                coldir = meta[idxim].get('ImageColumnDir')
                dotrowdir = float(head[idxim].phase_dir[0]) * float(rowdir[0]) + float(head[idxim].phase_dir[1]) * float(rowdir[1]) + float(head[0].phase_dir[2]) * float(rowdir[2])
                dotrowdir = abs(dotrowdir)
                dotcoldir = float(head[idxim].phase_dir[0]) * float(coldir[0]) + float(head[idxim].phase_dir[1]) * float(coldir[1]) + float(head[0].phase_dir[2]) * float(coldir[2])
                dotcoldir = abs(dotcoldir)

                logging.info("%f vs %f ", dotrowdir, dotcoldir)
                if dotrowdir > dotcoldir:
                    NeedTranspose = True
                else:
                    NeedTranspose = False

                if NeedTranspose:
                    logging.info('Performing Transpose')
                    lr_image = np.transpose(lr_image)

                lr_image, minval, maxval = PercentileRescaler(lr_image)
                hr_tensor = image2tensor(lr_image).half()
                hr_tensor = hr_tensor.unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    sr_tensor = generator(hr_tensor)
                    sr_tensor = sr_tensor.squeeze()
                    img_a = np.array(sr_tensor.to('cpu'), np.float32, copy=False)

                    sr_tensor2 = generator2(hr_tensor)
                    sr_tensor2 = sr_tensor2.squeeze()
                    img_b = np.array(sr_tensor2.to('cpu'), np.float32, copy=False)
                    img=(img_a+img_b)/2.0

                    img = np.clip(img, 0, 1)
                    if NeedTranspose:
                        img = np.transpose(img)

                    data[:, :, Dim3, Dim4, Dim5] = RestoreRescaler(np.array(img), minval, maxval)

                idxim = idxim + 1
                logging.info("Processed using SY_recon %d / %d, minval: %f, maxval : %f ", idxim, total, minval, maxval)

    # Normalize and convert to int16
    data = data.astype(np.float32)

#    data *= 32767 / data.max()
    data = np.floor(data)
    data = data.astype(np.int16)
    # Reformat data from [row col z cha img] back to [x y z cha img] before sending back to client

    data = data.transpose((1, 0, 2, 3, 4))

    currentSeries = 0

    total_phase_count, total_slice_count = head[-1].phase+1, head[-1].slice+1

    blood_pool_buff = np.array([[None] * total_phase_count] * total_slice_count)
    myocardium_buff = np.array([[None] * total_phase_count] * total_slice_count)

    logging.info("Begining segmentation...")

    # Re-slice back into 2D images
    imagesOut = [None] * data.shape[-1]
    imagesOutROI = [None] * data.shape[-1] # Secondary list to hold the images 
    for iImg in range(data.shape[-1]):
        # Create new MRD instance for the inverted image
        # NOTE: from_array() takes input data as [x y z coil], which is
        # different than the internal representation in the "data" field as
        # [coil z y x].  However, we already transposed this data when
        # extracting it earlier.
        imagesOut[iImg] = ismrmrd.Image.from_array(data[...,iImg])
        data_type = imagesOut[iImg].data_type

        imagesOutROI[iImg] = ismrmrd.Image.from_array(data[...,iImg])
        data_type = imagesOutROI[iImg].data_type

        # Create a copy of the original fixed header and update the data_type
        # (we changed it to int16 from all other types)
        oldHeader = head[iImg]
        oldHeader.data_type = data_type

        # Unused example, as images are grouped by series before being passed into this function now
        # oldHeader.image_series_index = currentSeries

        # Increment series number when flag detected (i.e. follow ICE logic for splitting series)
        if mrdhelper.get_meta_value(meta[iImg], 'IceMiniHead') is not None:
            if mrdhelper.extract_minihead_bool_param(base64.b64decode(meta[iImg]['IceMiniHead']).decode('utf-8'), 'BIsSeriesEnd') is True:
                currentSeries += 1

        imagesOut[iImg].setHead(oldHeader)

        # ROI Images will be a seperate series
        oldHeader.image_series_index = (oldHeader.image_series_index + 1)
        imagesOutROI[iImg].setHead(oldHeader)

        # Create a copy of the original ISMRMRD Meta attributes and update
        tmpMeta = meta[iImg]
        minval=np.min(imagesOut[iImg].data)
        maxval=np.max(imagesOut[iImg].data)

        center= (minval+maxval)/2
        window= int(center*2)
        tmpMeta['DataRole']                       = 'Image'
        tmpMeta['ImageProcessingHistory']         = ['PYTHON', 'GAN']
        tmpMeta['WindowCenter']                   = str(center)
        tmpMeta['WindowWidth']                    = str(window)
        tmpMeta['SequenceDescriptionAdditional']  = '- Reconed'
        tmpMeta['Keep_image_geometry']            = 1
        tmpMeta['SiemensDicom_ContentQualification'] = ['string', 'PRODUCT']

        # Add image orientation directions to MetaAttributes if not already present
        if tmpMeta.get('ImageRowDir') is None:
            tmpMeta['ImageRowDir'] = ["{:.18f}".format(oldHeader.read_dir[0]), "{:.18f}".format(oldHeader.read_dir[1]), "{:.18f}".format(oldHeader.read_dir[2])]

        if tmpMeta.get('ImageColumnDir') is None:
            tmpMeta['ImageColumnDir'] = ["{:.18f}".format(oldHeader.phase_dir[0]), "{:.18f}".format(oldHeader.phase_dir[1]), "{:.18f}".format(oldHeader.phase_dir[2])]

        roi_list, image_mask = create_roi_from_contour(data[:,:,0,0,iImg])

        blood_pool_voxels_count = np.squeeze(image_mask==3).sum(axis=(0,1))
        myocardium_voxels_count = np.squeeze(image_mask==2).sum(axis=(0,1))

        blood_pool_buff[head[iImg].slice, head[iImg].phase] = blood_pool_voxels_count
        myocardium_buff[head[iImg].slice, head[iImg].phase] = myocardium_voxels_count

        metaXml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
        logging.debug("Image data has %d elements", imagesOut[iImg].data.size)

        imagesOut[iImg].attribute_string = metaXml

        # Set ROI field for second set of images
        for index, roi in enumerate(roi_list):
            # roi_tag = 'ROI_' + str(iImg) + str(index)
            roi_tag = 'ROI_' + str(head[iImg].slice) + '_' + str(head[iImg].phase) + '_' + str(index)
            logging.debug("ROI Tag: %s", roi_tag)
            tmpMeta[roi_tag] = roi

        # Assign seperate description for segmented images
        tmpMeta['SequenceDescriptionAdditional']  = '- Segmented'

        metaXml_with_roi = tmpMeta.serialize()

        imagesOutROI[iImg].attribute_string = metaXml_with_roi

    logging.info("Segmentation complete...")

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

    logging.info('SLICE THICKNESS: %f', slice_thickness)

    voxel_volume = (pixel_spacing_x * pixel_spacing_y * slice_thickness)/1000

    lvef_figure, edv, esv, sv, ef = create_lvef_figure(blood_pool_buff, myocardium_buff, voxel_volume)

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
    
    lvef_figure_meta['EDV'] = edv
    lvef_figure_meta['ESV'] = esv
    lvef_figure_meta['SV'] = sv
    lvef_figure_meta['EF'] = ef
    
    # lvef_figure_meta['ImageRowDir'] = ["{:.18f}".format(oldHeader.read_dir[0]), "{:.18f}".format(oldHeader.read_dir[1]), "{:.18f}".format(oldHeader.read_dir[2])]
    # lvef_figure_meta['ImageColumnDir'] = ["{:.18f}".format(oldHeader.phase_dir[0]), "{:.18f}".format(oldHeader.phase_dir[1]), "{:.18f}".format(oldHeader.phase_dir[2])]
    # lvef_figure_meta['ImageSliceNormDir'] = ["{:.18f}".format(oldHeader.slice_dir[0]), "{:.18f}".format(oldHeader.slice_dir[1]), "{:.18f}".format(oldHeader.slice_dir[2])]
    
    lvef_figure_meta['ImageRowDir'] = mrdhelper.get_meta_value(ismrmrd.Meta.deserialize(images[-1].attribute_string), 'ImageRowDir')
    lvef_figure_meta['ImageColumnDir'] = mrdhelper.get_meta_value(ismrmrd.Meta.deserialize(images[-1].attribute_string), 'ImageColumnDir')
    lvef_figure_meta['ImageSliceNormDir'] = mrdhelper.get_meta_value(ismrmrd.Meta.deserialize(images[-1].attribute_string), 'ImageSliceNormDir')

    # When sending figure we can bypass all normalization functors in the ICE pipeline
    # lvef_figure_meta['InternalSend'] = 1
    lvef_figure_meta['DirectSend'] = 1
    
    metaXml = lvef_figure_meta.serialize()
    lvef_figure.attribute_string = metaXml

    # Append images with ROI to output array
    imagesOut += imagesOutROI

    # Append figure to output array
    imagesOut.append(lvef_figure)

    return imagesOut

def create_roi_from_contour(img):
    img = img[..., np.newaxis, np.newaxis]
    image_mask = get_mask(img, netS)
    # image_mask = img > np.percentile(img.ravel(), 80)
    # image_mask = np.rot90(image_mask, 3)
    # image_mask = np.flipud(image_mask)
    # image_mask = np.fliplr(image_mask)
    image_mask = np.transpose(image_mask)

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

    strove_volume = int(np.round((edv_blood_pool_total-esv_blood_pool_total)))
    ejection_fraction = int(np.round((strove_volume / edv_blood_pool_total)*100))
    edv_blood_pool_total = int(np.round(edv_blood_pool_total))
    esv_blood_pool_total = int(np.round(esv_blood_pool_total))

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

    return lvef_figure, edv_blood_pool_total, esv_blood_pool_total, strove_volume, ejection_fraction
