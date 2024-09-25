import albumentations as A


def get_training_augmentation(path_yaml_config: str):
    transforms_train = []
    loaded_transform = A.load(filepath=path_yaml_config, data_format='yaml')
    transforms_list = loaded_transform.transforms
    transforms_train.extend(transforms_list
    )

    return A.Compose(transforms_train)



