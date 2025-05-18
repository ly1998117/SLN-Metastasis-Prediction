import os.path
from collections import OrderedDict

import nibabel
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from monai.transforms import LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, SaveImaged, Compose, \
    CropForegroundd, generate_spatial_bounding_box, ScaleIntensityd, MapTransform
from monai.data import Dataset, DataLoader
from monai.utils import convert_data_type
from skimage.exposure import match_histograms

torch.multiprocessing.set_sharing_strategy('file_system')


class Round(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]
            image[image >= 0.2] = 1
            image[image < 0.2] = 0
            d[key] = image
        return d


class N4BiasFieldCorrectiond(MapTransform):
    """
    自定义的 N4 偏场校正转换，使用 SimpleITK 实现。
    """

    def __init__(
            self,
            keys,
            mask_key=None,
            copy_meta=True,
            allow_missing_keys=False,
    ):
        """
        初始化参数：
        - keys: 需要进行偏场校正的图像键列表。
        - mask_key: 可选，指定掩码的键，仅对感兴趣区域进行校正。
        - copy_meta: 是否复制元数据，默认值为 True。
        - allow_missing_keys: 是否允许缺少键，默认值为 False。
        """
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.copy_meta = copy_meta

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            # 将 NumPy 数组转换为 SimpleITK 图像
            sitk_image = sitk.GetImageFromArray(img[0])  # 假设图像是 [C, H, W, D]，取第一个通道
            sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)

            # 如果提供了掩码
            if self.mask_key and self.mask_key in d:
                mask = d[self.mask_key]
                sitk_mask = sitk.GetImageFromArray(mask[0].astype(np.uint8))
            else:
                # 使用 Otsu 阈值法生成掩码
                sitk_mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)

            # 设置 N4 偏场校正过滤器
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            # 执行校正
            corrected_image = corrector.Execute(sitk_image, sitk_mask)

            # 将校正后的图像转换回 NumPy 数组
            corrected_array = sitk.GetArrayFromImage(corrected_image)
            # 恢复形状并添加通道维度
            corrected_array = corrected_array[np.newaxis, ...]
            corrected_array = np.ascontiguousarray(corrected_array)
            d[key] = img.set_array(corrected_array)
        return d


class HistogramStandardization(MapTransform):
    def __init__(self, keys, reference, bins=256, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.reference = reference
        self.bins = bins

    def standardize(self, image):
        """
        Adjust the image histogram to match the reference.
        This example uses a basic cumulative histogram matching approach.

        Args:
            image (np.ndarray): The input image to standardize.

        Returns:
            np.ndarray: Image with matched histogram.
        """
        # Compute histogram of the image
        image_hist, image_bin_edges = np.histogram(image, bins=self.bins, range=(image.min(), image.max()),
                                                   density=True)
        image_cdf = np.cumsum(image_hist)

        # Create a mapping from the input image intensities to the reference histogram
        reference_cdf = np.linspace(0, 1, len(self.reference))
        mapping = np.interp(image_cdf, reference_cdf, self.reference)

        # Map the original image intensities using this mapping
        standardized_image = np.interp(image.ravel(), image_bin_edges[:-1], mapping)
        return standardized_image.reshape(image.shape)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]
            d[key] = self.standardize(image)
        return d


class HistogramMatchd(MapTransform):
    """
    Dictionary-based transform that matches the histogram of the image
    specified by `keys` to the reference image.
    """

    def __init__(self, keys, ref_image, allow_missing_keys=False):
        """
        Args:
            keys: List of keys to apply the transform.
            ref_image: Reference image to match histograms with.
            allow_missing_keys: If True, do not raise an exception if a key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.ref_image = np.expand_dims(nibabel.load(ref_image).get_fdata(), axis=0)

    def __call__(self, data):
        """
        Args:
            data: Dictionary containing the images.

        Returns:
            A dictionary with updated histogram-matched images.
        """
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = self._match_histogram(d[key])
        return d

    def _match_histogram(self, image):
        """
        Perform histogram matching between the input image and the reference image.

        Args:
            image (np.ndarray): Input image to be matched.

        Returns:
            np.ndarray: Histogram-matched image.
        """
        # Ensure both image and ref_image are in the same format

        # Perform histogram matching
        matched = match_histograms(np.asarray(image), np.asarray(self.ref_image), channel_axis=None)
        return image.set_array(matched)


class MyCropForegroundd(CropForegroundd):
    def __call__(self, data, lazy: bool | None = None):
        d = dict(data)
        bses, bees = [], []
        select_fn = self.cropper.select_fn
        if not isinstance(self.source_key, list):
            self.source_key = [self.source_key]
        if not isinstance(select_fn, list):
            select_fn = [select_fn]
        for source_key, fn in zip(self.source_key, select_fn):
            self.cropper.select_fn = fn
            box_start, box_end = self.cropper.compute_bounding_box(img=d[source_key])
            bses.append(box_start)
            bees.append(box_end)
        self.cropper.select_fn = select_fn
        bses, bees = np.array(bses), np.array(bees)
        box_start, box_end = bses.min(axis=0), bees.max(axis=0)
        if self.start_coord_key is not None:
            d[self.start_coord_key] = box_start  # type: ignore
        if self.end_coord_key is not None:
            d[self.end_coord_key] = box_end  # type: ignore

        lazy_ = self.lazy if lazy is None else lazy
        for key, m in self.key_iterator(d, self.mode):
            d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m, lazy=lazy_)
        return d


############################################# PREPROCESS #############################################
def select(ignore_margin=False):
    def _fn(x):
        x = (x - x.min()) / (x.max() - x.min()) * 255
        x = x > 30
        assert x.ndim == 4
        if ignore_margin:
            # margin = int(x.shape[2] * 5 / 32)
            margin = 1
            for i in range(1, 100):
                if x[..., -i, :].sum() == 0:
                    margin = i
            x[..., -margin:, :] = False
        return x

    return _fn


def select_breast(mode='square'):
    def _fn(x):
        mid = x.shape[1] // 2
        x = (x - x.min()) / (x.max() - x.min()) * 255
        _, box_end = generate_spatial_bounding_box(x[:, mid, ...], select_fn=lambda x: x > 40)
        box_end, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        box_end = box_end[0]
        indices = torch.zeros_like(x).bool()
        if mode == 'square':
            box_end = min(box_end, x.shape[2] - mid)
        elif mode == 'margin':
            box_end -= int(x.shape[1] * 5 / 32)
        else:
            raise ValueError(f"Unknown mode {mode}")
        indices[..., box_end:, :] = True
        return indices

    return _fn


def select_breast_part(x):
    assert x.sum() > 0, f"Empty mask {x.shape} {x.sum()} {x.meta['filename_or_obj']}"
    mid = x.shape[1] // 2
    indices = torch.zeros_like(x).bool()
    if x[:, :mid, ...].sum() > x[:, mid:, ...].sum():
        indices[:, :mid, ...] = True
    else:
        indices[:, mid:, ...] = True
    return indices


def get_data_paths(filepath='data/CSV/SLN_sec1_20240805_correct.xlsx'):
    df = pd.read_excel(filepath) if filepath.endswith('.xlsx') else pd.read_csv(filepath)
    pt = [
        # 'CENGXIAOLING_050Y_0005314447.nii.gz',
        # 'CHENYULIAN_042Y_0013461906.nii.gz',
        # 'ganwei_057Y_0019100566.nii.gz',
        # 'WANGQIN_038Y_0004653724.nii.gz',
        # 'XIEHUI_045Y_0004442320.nii.gz',
        # 'XIONGTONGYU_049Y_0002353596.nii.gz',
        # 'RENXUEMEI_048Y_0019056165.nii.gz'

    ]
    if len(pt) > 0:
        data_paths = df['path_image'][df['path_image'].map(lambda x: os.path.basename(x) in pt)]
        label_paths = df['path_mask'][df['path_mask'].map(lambda x: os.path.basename(x) in pt)]
    else:
        data_paths = df['path_image']
        label_paths = df['path_mask']
    return data_paths, label_paths


def preprocess(data_root_dir='data/raw_data', mask_root_dir='data/mask', output_dir='data/nii',
               filepath='data/CSV/SLN_correct.xlsx', pixdim=(1, 1, 1), n4BiasFieldCorrection=False,
               histogramStandard=False, foreground=False, breast=False, crop=False, maxv=None, output_dtype=np.int16):
    df = pd.read_excel(filepath) if filepath.endswith('.xlsx') else pd.read_csv(filepath)
    df['path_image'] = df['path_image'].map(lambda x: x.replace(data_root_dir, output_dir + '/image'))
    df['path_mask'] = df['path_mask'].map(lambda x: x.replace(mask_root_dir, output_dir + '/mask'))

    os.makedirs(f'{output_dir}/CSV', exist_ok=True)
    df.to_csv(f'{output_dir}/CSV/data.csv', index=False)
    data_paths, label_paths = get_data_paths(filepath)
    transform = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
    if foreground:
        transform.append(
            CropForegroundd(['image', 'label'], source_key='image', select_fn=select(ignore_margin=True)))

    if n4BiasFieldCorrection:
        transform.append(N4BiasFieldCorrectiond(keys=["image"], mask_key="label"))
    if pixdim is not None:
        transform.append(Spacingd(keys=["image", "label"], pixdim=pixdim, mode=(3, "nearest")))
    transform.append(Round(keys=["label"]))
    if histogramStandard:
        # transform.append(HistogramStandardization(keys=["image"], reference=[0, 10, 100, 1000, 8000], bins=1024))
        transform.append(HistogramMatchd(keys=["image"],
                                         ref_image='/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/raw_data/image/BAOLI_059Y_0013340495.nii.gz'))

    if breast:
        transform.append(
            MyCropForegroundd(['image', 'label'], source_key=['image', 'label'],
                              select_fn=[select_breast('margin'), lambda x: x > 0.2]))
        transform.append(
            CropForegroundd(['image', 'label'], source_key='image', select_fn=select(ignore_margin=False)))
    if crop:
        transform.append(
            MyCropForegroundd(['image', 'label'], source_key=['label', 'label'],
                              select_fn=[select_breast_part, lambda x: x > 0.2]))
        transform.append(
            CropForegroundd(['image', 'label'], source_key='image', select_fn=select(ignore_margin=False)))
        # transform.append(ROICenterCropd(['image', 'label'], source_key='label', spatial_size=96))

    if maxv is not None:
        transform.append(ScaleIntensityd(keys=["image"], minv=0, maxv=maxv))

    transform.extend([SaveImaged(keys=["image"], output_dir=output_dir + '/image', separate_folder=False,
                                 data_root_dir=data_root_dir, output_postfix='', output_dtype=output_dtype),
                      SaveImaged(keys=["label"], output_dir=output_dir + '/mask', separate_folder=False,
                                 data_root_dir=mask_root_dir,
                                 output_postfix='',
                                 output_dtype=np.int16)])
    dataset = Dataset(
        data=[{'image': data_path, 'label': label_path} for data_path, label_path
              in zip(data_paths, label_paths)], transform=Compose(transform)
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=16, shuffle=False)
    for d in loader:
        pass


############################################# APP #############################################
def check_affine(filepath=''):
    data = pd.read_csv(filepath)
    for data_path, mask_path in zip(data['path_image'], data['path_mask']):
        data = nibabel.load(data_path)
        mask = nibabel.load(mask_path)
        assert data.shape == mask.shape, f"Shape not equal {data_path} {mask_path}"
        if (data.affine - mask.affine).sum() > 0:
            print(f"Affine {(data.affine - mask.affine).sum()} not equal {data_path} {mask_path}")
            mask_d = mask.get_fdata().astype(np.int16)
            assert mask_d.sum() > 0, f"Empty mask {mask_path}"
            nibabel.save(nibabel.Nifti1Image(mask_d, data.affine), mask_path)


def breast(dataset='SLN_internal', postfix=''):
    preprocess(output_dir=f'{dataset}/breast{postfix}', pixdim=(1, 1, 1),
               filepath=f'{dataset}/radiomics_n4{postfix}/CSV/data.csv',
               data_root_dir=f'{dataset}/radiomics_n4{postfix}/image',
               mask_root_dir=f'{dataset}/radiomics_n4{postfix}/mask',
               foreground=False, breast=True, crop=False)


def crop(dataset='SLN_internal', postfix=''):
    preprocess(output_dir=f'{dataset}/cropped{postfix}', pixdim=None,
               filepath=f'{dataset}/breast{postfix}/CSV/data.csv',
               data_root_dir=f'{dataset}/breast{postfix}/image',
               mask_root_dir=f'{dataset}/breast{postfix}/mask',
               foreground=False, breast=False, crop=True)


def outline_process():
    from .utils import crop_nii, resize_nii
    data_paths, label_paths = get_data_paths()
    for data_path, label_path in zip(data_paths, label_paths):
        img, mask = crop_nii(data_path, label_path, roi_start=(0, 0, 187), roi_end=(None, None, None))
        img, mask = resize_nii(image=img, mask=mask, spatial_size=(-1, -1, 374))
        os.makedirs(os.path.dirname(data_path.replace('breast_data/', 'breast_data2/')), exist_ok=True)
        os.makedirs(os.path.dirname(label_path.replace('mask/', 'mask2/')), exist_ok=True)
        nibabel.save(img, data_path.replace('breast_data/', 'breast_data2/'))
        nibabel.save(mask, label_path.replace('mask/', 'mask2/'))


def for_radiomics():
    preprocess(output_dir='SLN_internal/cropped', pixdim=(1, 1, 1),
               filepath='SLN_internal/radiomics_n4/CSV/data.csv',
               data_root_dir='SLN_internal/radiomics_n4/image',
               mask_root_dir='SLN_internal/radiomics_n4/mask',
               foreground=True, breast=True, crop=True)


def n4_bias_data(dataset, postfix=''):
    preprocess(data_root_dir=f'{dataset}/raw_data/image',
               mask_root_dir=f'{dataset}/raw_data/mask{postfix}',
               output_dir=f'{dataset}/radiomics_n4{postfix}',
               filepath=f'{dataset}/raw_data/file/{dataset}_correct{postfix}.csv',
               pixdim=(0.625, 0.625, 1),
               n4BiasFieldCorrection=True,
               histogramStandard=True,
               foreground=True,
               breast=False,
               crop=False,
               output_dtype=np.int16)


# tian rong_054Y_0000069330
# zhao fu lan_46Y_0005797159
def internal():
    check_affine('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/raw_data/file/SLN_internal_correct.csv')
    # n4_bias_data(data_root_dir='SLN_internal/raw_data/image', mask_root_dir='SLN_internal/raw_data/mask',
    #              output_dir='SLN_internal/SLN_internal_n4',
    #              filepath='SLN_internal/raw_data/file/SLN_internal_correct.csv')
    # breast()
    crop()
    # for_radiomics()
    pass


def external():
    # check_affine('/data_smr/liuy/Project/BreastCancer/dataset/SLN_external/raw_data/file/SLN_external_correct.csv')
    # n4_bias_data('SLN_external')
    breast('SLN_external')
    crop('SLN_external')
    # for_radiomics()
    pass


def icc():
    check_affine('/data_smr/liuy/Project/BreastCancer/dataset/SLN_internal/raw_data/file/SLN_internal_correct_icc.csv')
    # n4_bias_data('SLN_internal', postfix='_icc')
    breast('SLN_internal', postfix='_icc')
    crop('SLN_internal', postfix='_icc')


if __name__ == "__main__":
    icc()
