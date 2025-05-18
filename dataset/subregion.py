import os

import SimpleITK as sitk
import pandas as pd
import nibabel as nib
import cv2
import numpy as np
import skimage.morphology as morphology
import mpire as mpi
import matplotlib.pyplot as plt
from skimage import measure
from functools import wraps
from scipy.stats import entropy as calc_entropy
from skimage.segmentation import slic
from sklearn.preprocessing import StandardScaler
import shutil


def multi_process(num_process):
    def decorator(func):
        @wraps(func)
        def wrapper(params):
            with mpi.WorkerPool(num_process) as pool:
                return pool.map(func, params, progress_bar=True)

        return wrapper

    return decorator


@multi_process(16)
def slic_supervoxels(image_path, mask_path, n_segment, output_dir):
    print(f'Start super voxel processing ... {image_path}, {mask_path}')
    output_path = os.path.join(output_dir, 'SuperVoxel', os.path.basename(mask_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path)
    affine = mask.affine
    mask = mask.get_fdata()
    img_normalized = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    tmp_mask = morphology.opening(mask, morphology.ball(1))

    try:
        slic_mask = slic(img_normalized, n_segment, compactness=10, mask=tmp_mask, start_label=1,
                         channel_axis=None)
        slic_mask = slic_mask.astype(np.uint16)
    except:
        print('something wrong, SV computation failed !!!')
        raise ValueError('SV computation failed !!!')

    mask_itk = nib.Nifti1Image(slic_mask, affine=affine)

    # save slic_mask
    nib.save(mask_itk, output_path)
    print(f'{os.path.basename(output_path)} is Done')
    return {
        'path_image': image_path,
        'path_mask': mask_path,
        'super_voxel': output_path
    }


@multi_process(16)
def subregion_split(image_path, mask_path, out_dir):
    region = nib.load(image_path).get_fdata()
    submask = nib.load(mask_path).get_fdata()
    affine = nib.load(mask_path).affine

    _, num = measure.label(submask, connectivity=3, return_num=True)  # meature label connected regions
    paths = []
    filename = os.path.basename(mask_path).split('.')[0]
    os.makedirs(os.path.join(out_dir, 'SVimage', filename), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'SVmask', filename), exist_ok=True)

    print(f'Processing {mask_path} ...')
    for i in range(1, num + 1):
        sv_mask_split = np.zeros_like(submask)
        sv_mask_split[submask == i] = 1

        # ----- for SV mask split ------
        # print(mask.split('/')[-1].split('.')[-3]))

        svimage_path = os.path.join(out_dir, 'SVimage', filename, f'{i:03d}.nii.gz')
        svmask_path = os.path.join(out_dir, 'SVmask', filename, f'{i:03d}.nii.gz')
        nib.save(nib.Nifti1Image(sv_mask_split, affine=affine), svmask_path)

        # ----- for SV volume split ------
        svimage_split = np.zeros_like(region)
        svimage_split[submask == i] = region[submask == i]
        nib.save(nib.Nifti1Image(svimage_split, affine=affine), svimage_path)
        paths.append({
            'path_image': image_path,
            'svid': i,
            'svimage': svimage_path,
            'svmask': svmask_path
        })
    return pd.DataFrame(paths)


###########################################################################################
import numpy as np
import pandas as pd
import glob
import os

from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

'''
return the K value and covariance_type from the best GMM model according to the BIC
'''


def get_K(feature, K_num):
    if np.unique(feature).shape[0] <= 1:
        # 所有值都一样，无法聚类
        return 1, np.zeros_like(feature, dtype=int)
    feature = preprocessing.scale(feature)
    lowest_bic = np.infty
    n_components_range = range(1, min(K_num, feature.shape[0]) + 1)
    # cv_types = ["spherical", "tied", "diag", "full"]
    best_gmm = None
    best_labels = None
    for n_components in n_components_range:
        gmm = GaussianMixture(
            n_components=n_components, covariance_type='full', random_state=0
        )
        gmm.fit(feature)
        bic = gmm.bic(feature)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm
            best_labels = gmm.predict(feature)

    return best_gmm.n_components, best_labels


def assign_K_value2features(patient_features, K_num):
    K_list = []
    for feature_idx in range(patient_features.shape[1]):
        X_train = np.array(patient_features[:, feature_idx]).reshape(-1, 1)
        k, best_labels = get_K(X_train, K_num)
        K_list.append(k)
    return K_list


def compute_entropy_vector(values, bins=20):
    if len(np.unique(values)) <= 1:
        return 0.0  # 熵为0（无信息量）
    hist, _ = np.histogram(values, bins=bins, density=False)
    hist = hist[hist > 0]
    if hist.sum() == 0:
        return 0.0
    hist = hist / hist.sum()
    return calc_entropy(hist)


def region_feature_fusion_vectorized(features, K_num=6, reduction="mean", method="entropy"):
    """
    features: shape = (n_samples, n_features)
    返回: shape = (n_features,)
    """
    n_voxels, n_features = features.shape
    result = []

    for feat_idx in range(n_features):
        f = features[:, feat_idx].reshape(-1, 1)
        K, labels = get_K(f, K_num)

        df = pd.DataFrame({'value': f.squeeze(), 'label': labels.squeeze()})
        if method == 'entropy':
            values = df.groupby('label')['value'].apply(lambda x: compute_entropy_vector(x.values)).values
        else:
            values = df.groupby('label')['value'].mean().values

        weights = df['label'].value_counts(normalize=True).sort_index().values

        if reduction == "sum":
            fused = values.sum()
        elif reduction == "mean":
            fused = values.mean()
        elif reduction == "weighted":
            fused = np.sum(weights * values)
        else:
            raise ValueError("method must be one of 'sum', 'mean', 'weighted', 'entropy'")

        result.append(fused)

    return np.array(result)


@multi_process(32)
def get_edfeat(patient_info, patient_features, K_num):
    K = assign_K_value2features(patient_features.to_numpy(), K_num)
    try:
        patient_info.update({k: v for k, v in zip(patient_features.columns, K)})
    except:
        print('Error: someting wrong about K')
    return patient_info


@multi_process(32)
def get_ed_region_fusion(patient_info, patient_features, K_num, reduction="mean", method="entropy"):
    feat = region_feature_fusion_vectorized(patient_features.to_numpy(), K_num, reduction=reduction, method=method)
    try:
        patient_info.update({k: v for k, v in zip(patient_features.columns, feat)})
    except:
        print('Error: someting wrong about K')
    return patient_info


###########################################################################################
class SubRegion:
    def __init__(self, datadir, n_segment=100):
        self.n_segment = n_segment
        self.datadir = datadir
        self.data = pd.read_csv(f'{datadir}/CSV/data.csv')
        rootdir = os.path.dirname(os.path.dirname(datadir))
        self.data['path_image'] = self.data['path_image'].map(lambda x: os.path.join(rootdir, x))
        self.data['path_mask'] = self.data['path_mask'].map(lambda x: os.path.join(rootdir, x))

    def sr_split(self):
        if not os.path.exists(f'{self.datadir}/CSV/super_voxel.csv'):
            params = []
            for image, mask in zip(self.data['path_image'], self.data['path_mask']):
                params.append((image, mask, self.n_segment, self.datadir))
            df = pd.DataFrame(slic_supervoxels(params))
            df.to_csv(f'{self.datadir}/CSV/super_voxel.csv', index=False)
        else:
            df = pd.read_csv(f'{self.datadir}/CSV/super_voxel.csv')

        if not os.path.exists(f'{self.datadir}/CSV/SVmask.csv'):
            params = []
            for image, mask in zip(df['path_image'], df['super_voxel']):
                params.append((image, mask, self.datadir))
            df = pd.concat(subregion_split(params)).reset_index(drop=True)
            df.to_csv(f'{self.datadir}/CSV/SVmask.csv', index=False)

    def sr_ed(self, features, feat_list=None, k_num=5):
        if os.path.exists(f'{self.datadir}/CSV/SV_ED_features.csv'):
            return pd.read_csv(f'{self.datadir}/CSV/SV_ED_features.csv')
        if feat_list is None:
            feat_list = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
        exclude_list = []
        include_list = []

        for c_names in features.columns.to_list():
            if any([feature_type in c_names for feature_type in feat_list]):
                include_list.append(c_names)
            else:
                exclude_list.append(c_names)
        exclude_list.remove('svid')
        exclude_list.remove('svimage')
        exclude_list.remove('svmask')
        params = []
        features['group'] = features['path_image']
        features.groupby('group').apply(lambda x: params.append(
            (x.reset_index()[exclude_list].iloc[0].to_dict(), x[include_list], k_num)
        ))
        result = pd.DataFrame(get_edfeat(params))
        result.to_csv(f'{self.datadir}/CSV/SV_ED_features.csv', index=False)
        return result

    def sr_ed_fusion(self, features, feat_list=None, k_num=5, reduction="mean", method="entropy"):
        path = f'{self.datadir}/CSV/SV_ED_features_{method}_{reduction}.csv'
        if os.path.exists(path):
            return pd.read_csv(path)
        if feat_list is None:
            feat_list = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
        exclude_list = []
        include_list = []

        for c_names in features.columns.to_list():
            if any([feature_type in c_names for feature_type in feat_list]):
                include_list.append(c_names)
            else:
                exclude_list.append(c_names)
        exclude_list.remove('svid')
        exclude_list.remove('svimage')
        exclude_list.remove('svmask')
        params = []
        features['group'] = features['path_image']
        features.groupby('group').apply(lambda x: params.append(
            (x.reset_index()[exclude_list].iloc[0].to_dict(), x[include_list], k_num, reduction, method)
        ))
        result = pd.DataFrame(get_ed_region_fusion(params))
        result.to_csv(path, index=False)
        return result


if __name__ == "__main__":
    sr = SubRegion(datadir='SLN_internal/cropped')
    sr.sr_split()
