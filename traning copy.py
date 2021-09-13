from torch.utils.data.dataset import random_split
from dataloader import DepthDatasetLoader
from torch.utils.data import Dataset, DataLoader
import torch
import logging
import sys
#from pathlib import Path

#import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

#from utils.data_loading import BasicDataset, CarvanaDataset
from dice_score import dice_loss
from evaluate import evaluate
from net import Net

#torch.cuda.set_per_process_memory_fraction(0.5, 0)
torch.cuda.empty_cache()

dd_object = DepthDatasetLoader(dataset_directory = "dataset/")
# print(dd_object[10].keys())
# print(dd_object[10]['rgb'].shape)
#print(dd_object[192]['filenames'])
#dd_object[10]
print(len(dd_object))
# train_set, val_set = random_split(dd_object, [500, 100],generator=torch.Generator().manual_seed(42))


# train_dataloader = DataLoader(train_set, batch_size=7, shuffle=True, num_workers=1, drop_last=True)
# val_dataloader = DataLoader(val_set, batch_size=7, shuffle=True, num_workers=1, drop_last=True)

# for sub_iteration, minibatch in enumerate(train_dataloader):
#     print(sub_iteration)
#     print(minibatch['rgb'].shape, minibatch['filenames'])
#     break

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # # 2. Split into train / validation partitions
    n_val = 100
    n_train = 500
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_set, val_set = random_split(dd_object, [n_train, n_val],generator=torch.Generator().manual_seed(0))
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        # with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch in train_loader:
            images = batch['org']
            true_masks = batch['rgb']

            # assert images.shape[1] == net.n_channels, \
            #     f'Network has been defined with {net.n_channels} input channels, ' \
            #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
            #     'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=amp):
                images = images/255.0
                masks_pred = net(images)
                true_masks = true_masks / 255.0
                #print('mask_pred',masks_pred)
                #print('true_mask',true_masks.size())
                loss = criterion(masks_pred.float(), true_masks.float())
                print('loss',loss)
                # loss = criterion(masks_pred, true_masks) \
                #         + dice_loss(F.softmax(masks_pred, dim=1).float(),
                #                     F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                #                     multiclass=True)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            #pbar.update(images.shape[0])
            global_step += 1
            epoch_loss += loss.item()
            print(loss.item())
            print(global_step)
            print('EPOCH', epoch)
            # experiment.log({
            #     'train loss': loss.item(),
            #     'step': global_step,
            #     'epoch': epoch
            # })
            # pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round
            #if global_step % (n_train // (10 * batch_size)) == 0:
                # histograms = {}
                # for tag, value in net.named_parameters():
                #     tag = tag.replace('/', '.')
                #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                # val_score = evaluate(net, val_loader, device)
                # scheduler.step(val_score)

                # logging.info('Validation Dice score: {}'.format(val_score))
                # experiment.log({
                #     'learning rate': optimizer.param_groups[0]['lr'],
                #     'validation Dice': val_score,
                #     'images': wandb.Image(images[0].cpu()),
                #     'masks': {
                #         'true': wandb.Image(true_masks[0].float().cpu()),
                #         'pred': wandb.Image(torch.softmax(masks_pred, dim=1)[0].float().cpu()),
                #     },
                #     'step': global_step,
                #     'epoch': epoch,
                #     **histograms
                # })
            del loss, images, masks_pred
            torch.cuda.empty_cache()
        if save_checkpoint:
            #Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str('dir_checkpoint/'+'checkpoint_epoch{}.pth'.format(epoch + 1)))
            # logging.info(f'Checkpoint {epoch + 1} saved!')


if __name__ == '__main__':
    #args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet()

    # logging.info(f'Network:\n'
    #              f'\t{net.n_channels} input channels\n'
    #              f'\t{net.n_classes} output channels (classes)\n'
    #              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=4,
                  batch_size=5,
                  learning_rate=0.0001,
                  device=device,
                  img_scale=0.5,
                  val_percent=100 / 100,
                  amp=False)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)