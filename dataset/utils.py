import os.path
import re
import pandas as pd
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path


def statistic_mean_std(dataset, num_workers=4):
    """
    Calculate the mean and std of the dataset.
    Crop metatensor([463.0286]) metatensor([567.8478])
    No crop metatensor([315.2431]) metatensor([506.0428])
    """
    data_loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False)
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in tqdm(data_loader):
        data = data["image"]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return mean, std


def statistic_hist():
    all_landmarks = []
    data = pd.read_excel('data/CSV/SLN_correct.xlsx')
    for data_path in data['path_data']:
        data = nib.load(data_path).get_fdata().astype(np.int16)
        print(np.percentile(data, [1, 10, 50, 90, 99]))
        landmarks = np.percentile(data, [1, 10, 50, 90, 99])
        all_landmarks.append(landmarks)
    standard_landmarks = np.mean(all_landmarks, axis=0)
    print(standard_landmarks)


def read_nii(x):
    """
    Read the nii file.
    """
    return nib.load(x) if isinstance(x, str) else x


def crop_nii(image, mask, roi_start=None, roi_end=None):
    """
    Crop the nii file based on the mask.
    """
    image = read_nii(image)
    mask = read_nii(mask)
    image_data = image.get_fdata()
    mask_data = mask.get_fdata()
    image_data = image_data[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1], roi_start[2]:roi_end[2]]
    mask_data = mask_data[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1], roi_start[2]:roi_end[2]]
    return nib.Nifti1Image(image_data, image.affine), nib.Nifti1Image(mask_data, mask.affine)


def resize_nii(image, mask, spatial_size):
    """
    Resize the nii file based on the spatial size.
    """
    from monai.transforms import Resized
    resizer = Resized(keys=["image", "mask"], spatial_size=spatial_size, mode=("bilinear", "nearest"))
    image = read_nii(image)
    mask = read_nii(mask)
    image_data: np.array = image.get_fdata()
    mask_data = mask.get_fdata()
    data = resizer({'image': np.expand_dims(image_data, 0), 'mask': np.expand_dims(mask_data, 0)})
    image_data = data['image'][0].numpy().astype(np.int16)
    mask_data = data['mask'][0].numpy().astype(np.int16)
    return nib.Nifti1Image(image_data, image.affine), nib.Nifti1Image(mask_data, mask.affine)


def excel_process(file):
    xlsx = pd.read_excel(file, dtype=str)
    xlsx['pos'] = xlsx[['Name', 'ID']].apply(
        lambda x: 'left' if ('左' in x['Name'] or 'left' in x['Name'] or '左' in x['ID'] or 'left' in x['ID']) else (
            'right' if ('右' in x['Name'] or 'right' in x['Name'] or '右' in x['ID'] or 'right' in x['ID']) else None),
        axis=1)
    xlsx['id'] = xlsx['ID'].map(lambda x: x.split('_')[0].upper().strip())
    return xlsx


def generate_paths(datadir='data/raw_data', data_file='data/mask/SLN_sec1_20240805.xlsx', correct=dict(), postfix=''):
    def _load_data(data_path):
        data = []
        for d in Path(data_path).iterdir():
            if 'nii' in d.name:
                did = d.name.split('_')[-1].split('.nii')[0]
                if 'left' in d.name.lower():
                    pos = 'left'
                elif 'right' in d.name.lower():
                    pos = 'right'
                else:
                    pos = None

                if pos is not None:
                    did = d.name.split('_')[-2]
                if 'Y' in did and 'S' not in did:
                    did = did.split('Y')[-1]
                data.append({
                    'name': d.name.split('_')[0],
                    'id': did,
                    'pos': pos,
                    'filename': d.name.split('.nii')[0],
                    'path': d.__str__(),
                })
        data = pd.DataFrame(data).drop_duplicates(subset='filename')
        print(data[['filename']].drop_duplicates().__len__())
        return data

    image = _load_data(f'{datadir}/image')
    # mask = _load_data(f'{datadir}/mask')
    mask = _load_data(f'{datadir}/mask_icc')
    data = mask.merge(image, on=['name', 'id', 'pos', 'filename'], suffixes=('_mask', '_image'))
    data['id'] = data['id'].map(lambda x: x.strip().upper())
    print(mask[~mask['id'].isin(image['id'])])
    print(image[~image['id'].isin(mask['id'])])

    mask_xlsx = excel_process(data_file)

    for k, v in correct.items():
        mask_xlsx.loc[mask_xlsx['id'] == k, 'id'] = v
    mask_xlsx.loc[mask_xlsx['id'] == '0020289959', 'pos'] = None
    mask_xlsx.loc[mask_xlsx['id'] == '0032905122', 'pos'] = None

    merged = mask_xlsx.merge(data, on=['id', 'pos'])

    print(data[~data['id'].isin(merged['id'])])
    error_file = mask_xlsx[~mask_xlsx['id'].isin(merged['id'])]
    merged.to_csv(os.path.splitext(data_file)[0] + f'_correct{postfix}.csv', index=False)


def concat_pds(path1, path2, output_path):
    pd1 = pd.read_excel(path1)
    pd2 = pd.read_excel(path2)
    df = pd.concat([pd1, pd2])
    df['multi_label'] = df['SLN状态（0_无转移，1_转移）'] + df['SLN转移个数（0_转移≤2枚，1_转移≥3枚)']
    df.to_excel(output_path, index=False)


def load_cls_eachrepeat(dir_path, repeat=None, fold=None, epoch=None, stage=None):
    dfs = []
    for rps in os.listdir(dir_path):
        if 'repeat' not in rps:
            continue
        r = int(rps.split('_')[1])
        if repeat is not None and r != repeat:
            continue
        fold_path = os.path.join(dir_path, rps)
        df = load_cls_eachfold(fold_path, fold, epoch, stage, set_fn=lambda x: x.assign(repeat=r))
        dfs.append(df)
    dfs = pd.concat(dfs)
    return dfs


def load_cls_eachfold(dir_path, fold=None, epoch=None, stage=None, set_fn=None):
    if not isinstance(fold, list):
        fold = [fold]
    dfs = []
    for fold_name in os.listdir(dir_path):
        if 'fold' not in fold_name:
            continue
        f = int(fold_name.split('_')[1])
        if fold[0] is not None and f not in fold:
            continue
        fold_name = os.path.join(dir_path, fold_name, 'classification')
        for file in os.listdir(fold_name):
            if epoch is not None and str(epoch) not in file:
                continue
            if stage is not None and str(stage) not in file:
                continue
            s, e = os.path.splitext(file)[0].split('_')
            df = pd.read_csv(os.path.join(fold_name, file))
            df['stage'] = s
            df['epoch'] = e
            df['fold'] = f
            if set_fn is not None:
                df = set_fn(df)
            dfs.append(df)
    dfs = pd.concat(dfs)
    return dfs.rename(columns={'filename': 'path_data'})


def get_error_list(df, threshold=2, label=None):
    error = df.loc[df['pred'] != df['label']]
    error_num = error.groupby('path_data').count().reset_index()[['path_data', 'repeat']].rename(
        columns={'repeat': 'num'})
    error_num = error_num.loc[error_num['num'] >= threshold]
    error = error.loc[error['path_data'].isin(error_num['path_data'])][['path_data', 'pred', 'label']].drop_duplicates()
    if label is not None:
        error = error.loc[error['label'] == label]
    return error


def error():
    dir_path = '../run_5fold/breast_notest_ub_5fold_norm_e04'
    df = load_cls_eachrepeat(dir_path,
                             stage='val',
                             epoch=300)
    error = get_error_list(df, threshold=5)
    os.makedirs(f'../cleanlab/{os.path.basename(dir_path)}', exist_ok=True)
    error.to_csv(f'../cleanlab/{os.path.basename(dir_path)}/error.csv', index=False)

    error = get_error_list(df, threshold=3, label=0)
    os.makedirs(f'../cleanlab/{os.path.basename(dir_path)}', exist_ok=True)
    error.to_csv(f'../cleanlab/{os.path.basename(dir_path)}/error_0.csv', index=False)
    old = pd.read_csv(f'/data_smr/liuy/Project/BreastCancer/cleanlab/error_0.csv')
    old = pd.concat([old, error])
    old.to_csv(f'/data_smr/liuy/Project/BreastCancer/cleanlab/error_0.csv', index=False)


if __name__ == '__main__':
    # generate_paths(data_file='data/mask2/SLN_sec2_20240816.xlsx', correct={
    #     '0008131122': '00081311122',
    #     '0006641267': '00006641267',
    #     '00162894430016794097': '0016289443',
    #     '0037611679': '00376116789'
    # })
    # concat_pds('data/mask/SLN_sec1_20240805_correct.xlsx',
    #            'data/mask/SLN_sec2_20240816_correct.xlsx',
    #            'data/SLN_correct.xlsx')
    # dir_path = '../run_8fold/breast_notest_ub_repeat8fold_norm'
    # generate_paths(data_file='data/mask/SLN_20240909.xlsx', correct={
    #     '0008131122': '00081311122',
    #     '0006641267': '00006641267',
    #     '00162894430016794097': '0016289443',
    #     '0037611679': '00376116789',
    #     '0033306056': '0033306065',
    #     '0019445434': '0004020204',
    #     '0032181844': '0000234630',
    #     '0002723407': '002723407'
    # })
    ###################################################################################
    # generate_paths(datadir='SLN_internal/raw_data', data_file='SLN_internal/raw_data/file/SLN_internal.xlsx',
    #                correct={
    #                    '0037611679': '00376116789',
    #                    '0033306056': '0033306065',
    #                    '0019445434': '0004020204',
    #                    '0032181844': '0000234630',
    #                    '0002723407': '002723407',
    #                    '0008131122': '00081311122',
    #                    '0006641267': '00006641267',
    #                    '0000178826': '000178826',
    #                })

    # generate_paths(datadir='SLN_external/raw_data', data_file='SLN_external/raw_data/file/SLN_external.xlsx',
    #                correct={})
    generate_paths(datadir='SLN_internal/raw_data', data_file='SLN_internal/raw_data/file/SLN_internal.xlsx',
                   correct={
                       '0037611679': '00376116789',
                       '0033306056': '0033306065',
                       '0019445434': '0004020204',
                       '0032181844': '0000234630',
                       '0002723407': '002723407',
                       '0008131122': '00081311122',
                       '0006641267': '00006641267',
                       '0000178826': '000178826',
                   }, postfix='_icc')
    # error()
