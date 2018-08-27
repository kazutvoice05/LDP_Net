# coding: 'utf-8'

"""
LDP_Net
LDD_Transform

created by Kazunari on 2018/08/23 
"""

import cv2
import sys
import numpy as np

sys.path.append(".")

class LDDTransform(object):
    def __init__(self, localDepthDataset):
        self.mean_color = localDepthDataset.mean_color
        self.image_mean = localDepthDataset.image_mean
        self.image_stddev = localDepthDataset.image_stddev
        self.image_size = localDepthDataset.sun_image_size
        self.down_sampling_size = localDepthDataset.sun_down_sampling_size
        self.input_size = localDepthDataset.sun_input_size
        self.output_size = localDepthDataset.sun_output_size
        self.predicted_region = localDepthDataset.sun_predicted_region

        self.input_roi_size = (64, 64)
        self.class_id_size = localDepthDataset.get_class_id_size()

    """ Resize Image and Label Depth ( Original Size -> NYU Dataset Size (640 x 480) ) """
    def get_resized_data(self, img, depth, roi):
        """ Resize data to Image size(640 x 480) """
        resized_img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        resized_dpt = cv2.resize(depth, self.image_size, interpolation=cv2.INTER_NEAREST)

        original_size = img.shape[:2]
        size_diff_ratio = (self.image_size[1] / original_size[1],
                           self.image_size[0] / original_size[0])

        resized_roi = np.asarray([roi[0] * size_diff_ratio[0],
                                  roi[1] * size_diff_ratio[1],
                                  roi[2] * size_diff_ratio[0],
                                  roi[3] * size_diff_ratio[1]], dtype=np.float32)

        """ DownSampling resized data to resize to scale 1/2 """
        ds_img = cv2.resize(resized_img,
                            (self.down_sampling_size[1], self.down_sampling_size[0]),
                            interpolation=cv2.INTER_LINEAR)
        ds_dpt = cv2.resize(resized_dpt,
                            (self.down_sampling_size[1], self.down_sampling_size[0]),
                            interpolation=cv2.INTER_NEAREST)

        ds_roi = resized_roi / 2

        return ds_img, ds_dpt, ds_roi

    def get_predicted_region_data(self, img, depth, roi):
        y, x, h, w = self.predicted_region

        s_h = y + h
        s_w = x + w

        cropped_img = img[y:y + h, x:x + w]
        cropped_dpt = depth[y:y + h, x:x + w]

        cropped_roi = [max(0, roi[0] - x), max(0, roi[1] - y),
                       min(w - 1, roi[2] - x), min(h - 1, roi[3] - y)]

        return cropped_img, cropped_dpt, cropped_roi

    def get_roi_data(self, img, depth, pred_depth, roi):
        roi = np.asarray(np.floor(roi), dtype=np.int)

        dst_img = img[roi[1]:roi[3], roi[0]:roi[2]]
        dst_dpt = depth[roi[1]:roi[3], roi[0]:roi[2]]
        dst_pred_dpt = pred_depth[roi[1]:roi[3], roi[0]:roi[2]]

        return dst_img, dst_dpt, dst_pred_dpt

    def resize_to_input(self, img, depth, pred_depth):
        dst_img = cv2.resize(img, self.input_roi_size, interpolation=cv2.INTER_LINEAR)
        dst_depth = cv2.resize(depth, self.input_roi_size, interpolation=cv2.INTER_NEAREST)
        dst_pred_depth = cv2.resize(pred_depth, self.input_roi_size, interpolation=cv2.INTER_NEAREST)

        dst_img = np.asarray(dst_img, dtype=np.float32).transpose(2, 1, 0)

        dst_depth = np.asarray(dst_depth, dtype=np.float32).transpose(1, 0)
        dst_depth = np.expand_dims(dst_depth, axis=0)

        dst_pred_depth = np.asarray(dst_pred_depth, dtype=np.float32).transpose(1, 0)
        dst_pred_depth = np.expand_dims(dst_pred_depth, axis=0)

        return dst_img, dst_depth, dst_pred_depth

    # TODO : Normalize Image
    def __call__(self, in_data):
        roi, class_id, img, pred_depth, depth = in_data

        # Convert roi (u, v, w, h) -> local_region ( u1, v1, u2, v2 )
        roi_with_point = [roi[0], roi[1],
                          roi[0] + roi[2], roi[1] + roi[3]]

        resized_img, resized_depth, resized_roi = self.get_resized_data(img, depth, roi_with_point)
        cropped_img, cropped_depth, cropped_roi = self.get_predicted_region_data(resized_img, resized_depth,
                                                                                 resized_roi)
        roi_img, roi_depth, roi_pred_depth = self.get_roi_data(cropped_img, cropped_depth, pred_depth, cropped_roi)

        roi_img, roi_depth, roi_pred_depth = self.resize_to_input(roi_img, roi_depth, roi_pred_depth)

        # Create Mask
        eps = np.finfo(np.float32).eps
        mask = eps <= roi_depth

        def zscore(x, axis=None):
            xmean = x.mean(axis=axis, keepdims=True)
            xstd = np.std(x, axis=axis, keepdims=True)
            zscore = (x - xmean) / xstd
            return zscore

        # TODO : Implement Correct Normalization Function
        roi_img = zscore(roi_img)

        class_vector = np.zeros([self.class_id_size, self.input_roi_size[0], self.input_roi_size[1]], dtype=np.float32)
        class_vector[class_id, :, :] = 1

        #x = np.expand_dims(np.concatenate([roi_img, roi_pred_depth, class_vector], axis=0), axis=0)
        x = np.expand_dims(np.concatenate([roi_img, roi_pred_depth], axis=0), axis=0)
        t = np.expand_dims(roi_depth, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return x, t, mask
