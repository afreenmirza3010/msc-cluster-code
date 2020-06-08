

# importing all the necessary libraries for creating HDF5 files
import numpy as np
import h5py
from tqdm import tqdm


# used for visual(text based) progress during long running operations
def progress_bar(iterable):
    return tqdm(list(iterable))


# Extracting and printing the training(train_indices) , validation(val_indices) and
# testing(test_indices) from head and neck dataset containing 3D images

with h5py.File('../Dataset_3D_U_Net/Pnr.h5', 'r') as Patient_num:
    patient_id_indices = Patient_num['dataset1'][:].squeeze().astype(int)

with h5py.File('../Dataset_3D_U_Net/PnrVal.h5', 'r') as val_num:
    val_indices = val_num['dataset3'][:].squeeze().astype(int)
    val_new_indices = list(val_indices)
    print(val_new_indices)

with h5py.File('../Dataset_3D_U_Net/PnrTest.h5', 'r') as test_num:
    test_indices = test_num['dataset2'][:].squeeze().astype(int)
    test_new_indices = list(test_indices)
train_indices = list(set(patient_id_indices) - set(val_indices) - set(test_indices))
assert len(train_indices) == len(patient_id_indices) - len(val_indices) - len(test_indices)


# Method for combining CT and PET images and returns a 4D input image tensor with 2 channels
def extract_input_image(h5):
    ct_image = h5['imdata/CT']
    pt_image = h5['imdata/PT']
    final_image = np.stack([ct_image[:], pt_image[:]], axis=-1).squeeze()
    expected_shape = (173, 191, 265, 2)

    if final_image.shape != expected_shape:  # Check for expected shape after stacking CT and
        # PET images and reshaping it to desired shape if not correct
        print(f"{h5} has shape {final_image.shape}")
        try:
            print("Trying to reshape")
            final_image.reshape(expected_shape)
        except:
            print("Failed to reshape ")
            return None

    return final_image.astype('float32')


def extract_mask(h5):  # method extract_mask to combine tumor and nodes to create the final target
    # (mask) 4D image tensor
    tumor = h5['imdata/tumour'][:].squeeze()
    nodes = h5['imdata/nodes'][:].squeeze()
    return np.logical_or(tumor, nodes).squeeze().astype('float32')[..., np.newaxis]


# method that extract fold data into two separate lists for images and masks
# i.e. 4D images and masks for all the head and neck patients.
def extract_fold_data(fold_indices,
                      file_pattern='../Dataset_3D_U_NET/imdata/imdata_P{patient_id:03d}.mat',
                      verbose=True):
    if not verbose:
        def iterate(x):
            return x
    else:
        iterate = progress_bar

    images, masks = [], []
    for patient_id in iterate(fold_indices):
        with h5py.File(file_pattern.format(patient_id=int(patient_id)), 'r+') as h5:
            input_image = extract_input_image(h5)
            if input_image is not None:
                images.append(input_image)
                target = extract_mask(h5)
                masks.append(target)

    return images, masks


# method that create folds which are hdf5 containing dataset input,targets and
# patient numbers of all the 197 head and neck patients.
def create_fold(fold_indices, fold_num, filename, verbose=True):
    fold_indices = list(fold_indices)

    if verbose:
        print(f"Creating fold {fold_num}")
    images, masks = extract_fold_data(fold_indices)

    # The groups(the different folds) are created with dataset(input, targets, and patient number)
    with h5py.File(filename, 'a') as hdf:
        group = hdf.create_group(f'fold_{fold_num}')  # create a group with name eg. 'fold_0'
        group.create_dataset('input', data=images, compression='lzf', chunks=True)
        group.create_dataset('target', data=masks, compression='lzf', chunks=True)
        group.create_dataset('pat_num', data=fold_indices)

    return group


# methods to chunk groups into train ,validation and testing
def create_chunks(length, n):
    # looping till length
    for i in range(0, len(length), n):
        yield length[i:i + n]


patient_each_fold = 5  # choosing the number of patients in each fold
train_num_patients = len(train_indices)
val_num_patients = len(val_new_indices)
test_num_patients = len(test_new_indices)

# print(train_num_patients)
fold_num_train = int(np.ceil(train_num_patients / patient_each_fold))  # total no of folds
# for training patient's
fold_num_val = int(np.ceil(val_num_patients / patient_each_fold))  # total no of folds
# for validation patient's
fold_num_test = int(np.ceil(test_num_patients / patient_each_fold))  # total no of folds
# for testing patient's


file_name = "head_neck_new4.h5"  # the name of HDF5 file that will be generated with train,
# validation and test folds
num = 0


train_windows = list(create_chunks(train_indices, patient_each_fold)) #splitting training patients
# list with 5 patient each

for i in range(fold_num_train):
    train_fold = create_fold(fold_indices=train_windows[i],
                             fold_num=num, filename=file_name)
    print(train_fold)
    num += 1

# Create folds for val
val_windows = list(create_chunks(val_new_indices, patient_each_fold))
for s in range(fold_num_val):
    val_fold = create_fold(fold_indices=val_windows[s],
                           fold_num=num, filename=file_name)

    num += 1

# Create folds for test
test_windows = list(create_chunks(test_new_indices, patient_each_fold))
for t in range(fold_num_test):
    test_fold = create_fold(fold_indices=test_windows[t],
                            fold_num=num, filename=file_name)
    num += 1


