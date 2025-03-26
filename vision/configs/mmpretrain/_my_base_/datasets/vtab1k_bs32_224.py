# dataset settings
dataset_type = 'Vtab1K'
sub_dataset_name = 'cifar'

data_root = 'data/vtab-1k/'

data_preprocessor = dict(
    num_classes=100,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(224, 224),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(224, 224),
        interpolation='bicubic',
        backend='pillow'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train800val200.txt",
        sub_dataset_name=sub_dataset_name,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="test.txt",
        sub_dataset_name=sub_dataset_name,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1,))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
