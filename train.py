import datetime
import os
import re
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import download_weights, seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch

import csv
from utils.utils_metrics import fast_hist, per_class_iu, per_class_PA_Recall, per_class_Precision, per_Accuracy
# ==================== 数据收集函数（适配您的目录结构） ====================
def collect_from_category(category_dir):
    """
    从单个类别文件夹中收集所有图片和对应的 mask。
    category_dir 结构：
        category_dir/
            images/      (例如 Early_blight_1.JPG)
            masks/       (例如 Early_blight_1_pseudo.png 或 .jpg)
    """
    images_dir = os.path.join(category_dir, "images")
    masks_dir = os.path.join(category_dir, "masks")
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        return []

    pairs = []
    for img_file in os.listdir(images_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 提取数字编号（例如 "Early_blight_1.JPG" -> "1"）
        match = re.search(r'_(\d+)\.', img_file)
        if not match:
            print(f"警告：图片 {img_file} 命名格式不符合 _数字.，跳过")
            continue

        base_without_ext = re.sub(r'\.(jpg|jpeg|png)$', '', img_file, flags=re.I)

        # 尝试多种 mask 扩展名
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(masks_dir, f"{base_without_ext}_pseudo{ext}")
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            print(f"警告：未找到图片 {img_file} 对应的 mask")
            continue

        pairs.append((os.path.join(images_dir, img_file), mask_path))

    return pairs


def get_all_pairs(root_dir):
    """遍历 root_dir 下的每个类别子文件夹，收集所有 (img, mask) 对"""
    all_pairs = []
    for cat_name in os.listdir(root_dir):
        cat_path = os.path.join(root_dir, cat_name)
        if os.path.isdir(cat_path):
            pairs = collect_from_category(cat_path)
            all_pairs.extend(pairs)
    return all_pairs

def compute_metrics(model, dataloader, num_classes, device, input_shape):
    """在验证集上计算 mIoU, PA, mPA, Dice 等指标"""
    model.eval()
    hist = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for batch in dataloader:
            imgs, masks = batch[0], batch[1]
            if device.type == 'cuda':
                imgs = imgs.cuda()
                masks = masks.cuda()
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()
            for p, m in zip(preds, masks):
                hist += fast_hist(m.flatten(), p.flatten(), num_classes)

    # 原有指标
    iu = per_class_iu(hist)          # IoU per class
    mIoU = np.nanmean(iu)
    pa = per_Accuracy(hist)
    mpas = per_class_PA_Recall(hist)  # Recall per class
    mPA = np.nanmean(mpas)
    precisions = per_class_Precision(hist)

    # 新增: Dice 系数 (每个类)
    class_dice = np.zeros(num_classes)
    for c in range(num_classes):
        tp = hist[c, c]
        fp = hist[:, c].sum() - tp
        fn = hist[c, :].sum() - tp
        denominator = 2 * tp + fp + fn + 1e-12
        class_dice[c] = (2 * tp) / denominator
    mDice = np.nanmean(class_dice)

    return {
        'mIoU': mIoU,
        'PA': pa,
        'mPA': mPA,
        'class_IoU': iu,
        'class_PA': mpas,
        'class_Precision': precisions,
        'class_Dice': class_dice,
        'mDice': mDice
    }
# ==================== 主训练程序 ====================
if __name__ == "__main__":
    # -------------------------------#
    #   基础设置
    # -------------------------------#
    Cuda = False                     # 没有 GPU 时保持 False
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = False

    # -------------------------------#
    #   数据集路径（按 split_dataset 结构）
    # -------------------------------#
    dataset_root = "D:\seg\deeplabv3-plus-pytorch-main\split_dataset"
    train_root = os.path.join(dataset_root, "train")
    val_root   = os.path.join(dataset_root, "val")

    # 收集训练集和验证集的图片-mask对
    train_pairs = get_all_pairs(train_root)
    # early_pairs = [p for p in train_pairs if 'Early blight' in p[0]]
    # train_pairs = train_pairs + early_pairs * 1  # 只复制 1 份（即总共 2 倍）
    val_pairs   = get_all_pairs(val_root)

    train_img_paths = [p[0] for p in train_pairs]
    train_mask_paths = [p[1] for p in train_pairs]
    val_img_paths   = [p[0] for p in val_pairs]
    val_mask_paths  = [p[1] for p in val_pairs]

    num_train = len(train_img_paths)
    num_val   = len(val_img_paths)
    print(f"训练集图片数量：{num_train}")
    print(f"验证集图片数量：{num_val}")

    if num_train == 0 or num_val == 0:
        raise ValueError("没有找到训练集或验证集图片，请检查路径和文件命名。")

    # -------------------------------#
    #   模型参数
    # -------------------------------#
    num_classes = 4                     # 背景(0) + leaf(1) + Early blight(2) + Late blight(3)
    backbone = "mobilenet"              # "mobilenet" 或 "xception"
    pretrained = True
    model_path = "model_data/mobilenet_v2.pth"   # 预训练权重（可选）
    downsample_factor = 16
    input_shape = [256, 256]            # 可根据显存调整

    # -------------------------------#
    #   训练参数
    # -------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 50
    UnFreeze_Epoch = 1000
    Freeze_batch_size = 8
    Unfreeze_batch_size = 4
    Freeze_Train = True

    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 1e-4
    lr_decay_type = 'cos'

    save_period = 1
    save_dir = 'logs'
    eval_flag = False                    # 如需自动评估 mIoU 可设为 True，但需要实现 EvalCallback
    eval_period = 5
    VOCdevkit_path = None

    dice_loss = True
    focal_loss = True
    # cls_weights = np.array([0.1, 4.0, 1.0, 1.5], dtype=np.float32)
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 4

    seed_everything(seed)

    # 多卡/分布式设置（保持原样，一般单卡可不改）
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # 下载预训练权重
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    # 构建模型
    model = DeepLab(num_classes=num_classes, backbone=backbone,
                    downsample_factor=downsample_factor, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # LossHistory
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # 创建 CSV 文件用于保存每个 epoch 的指标
    csv_path = os.path.join(save_dir, 'training_metrics.csv')
    if local_rank == 0:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'PA', 'mPA', 'mIoU', 'mDice'])

    # fp16
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # 创建数据集和 DataLoader
    train_dataset = DeeplabDataset(train_img_paths, train_mask_paths, input_shape, num_classes, train=True)
    val_dataset   = DeeplabDataset(val_img_paths,   val_mask_paths,   input_shape, num_classes, train=False)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = Freeze_batch_size // ngpus_per_node if Freeze_Train else Unfreeze_batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                     sampler=train_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                         sampler=val_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

    # 学习率调整
    nbs = 16
    lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    if backbone == "xception":
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    }[optimizer_type]

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    # EvalCallback（如果需要验证 mIoU，请自行实现）
    eval_callback = None

    # 开始训练
    UnFreeze_flag = False
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size
            nbs = 16
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size
            if distributed:
                batch_size = batch_size // ngpus_per_node

            gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                             sampler=train_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
            gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate,
                                 sampler=val_sampler, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
            UnFreeze_flag = True

        if distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        train_loss, val_loss = fit_one_epoch(
            model_train, model, loss_history, eval_callback, optimizer, epoch,
            epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda,
            dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
            save_period, save_dir, local_rank
        )

        # # 每个 epoch 结束后计算验证集详细指标（仅在主进程）
        # if local_rank == 0:
        #     # 计算 mIoU, PA, mPA
        #     model_train.eval()
        #     metrics = compute_metrics(model_train, gen_val, num_classes, device, input_shape)
        #     current_lr = optimizer.param_groups[0]['lr']
        #
        #     # 追加到 CSV
        #     with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        #         writer = csv.writer(f)
        #         writer.writerow([
        #             epoch + 1,
        #             f"{current_lr:.6f}",
        #             f"{train_loss:.6f}",
        #             f"{val_loss:.6f}",
        #             f"{metrics['PA']:.6f}",
        #             f"{metrics['mPA']:.6f}",
        #             f"{metrics['mIoU']:.6f}"
        #         ])
        #     print(f"[Epoch {epoch + 1}] Metrics saved to {csv_path}")
        #     # 恢复训练模式
        #     model_train.train()
        if local_rank == 0:
            model_train.eval()
            metrics = compute_metrics(model_train, gen_val, num_classes, device, input_shape)
            current_lr = optimizer.param_groups[0]['lr']

            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    f"{current_lr:.6f}",
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{metrics['PA']:.6f}",
                    f"{metrics['mPA']:.6f}",
                    f"{metrics['mIoU']:.6f}",
                    f"{metrics['mDice']:.6f}"
                ])
            print(f"[Epoch {epoch + 1}] Metrics saved to {csv_path} (mDice = {metrics['mDice']:.4f})")
            model_train.train()
        if distributed:
            dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()