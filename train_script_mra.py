import argparse
from datetime import datetime
# import torchvision.models as models, ResNet50_Weights
# from models.Unet_3D.unet3d import UNet3D
from models.MobileViT_v3_3D.mobilevit_v3_trs_3d_1x6x6_same_slice import Mobilevit
import config_mra
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
from utils_ import load_nest_config
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from pytz import timezone
import wandb
import os


def save_model(epoch, validation_loss):
    loss_state_dict = val_loss_list[0].state_dict() if "UncertaintyLoss" in str(type(val_loss_list[0])) else None

    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir, exist_ok = True)

    torch.save({
            "epoch": epoch,
            "model_state_dict": architecture.state_dict(),
            "loss_state_dict": loss_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": validation_loss,
            "scheduler": scheduler.state_dict(),
        }, f"{model_save_dir}/model_{epoch}.pth"
    )


def calculate_train_loss(device):
    training_loss = 0
    for batch_idx, (inputs, *labels, label_weight, mask, file) in enumerate(train_loader):
        # print(inputs.shape)
        # inputs = F.interpolate(inputs, size=(32, 224, 224), mode='nearest')  # swinunetr, 추가된 코드
        # label_weight = F.interpolate(label_weight, size=(32, 224, 224), mode='nearest')  # swinunetr, 추가된 코드
        # mask = F.interpolate(mask, size=(32, 224, 224), mode='nearest')  # swinunetr, 추가된 코드
        inputs = F.interpolate(inputs, size=(28, 224, 224), mode='nearest')  # resize to SwinUNetR input size
        label_weight = F.interpolate(label_weight, size=(28, 224, 224), mode='nearest')  # resize to SwinUNetR input size
        mask = F.interpolate(mask, size=(28, 224, 224), mode='nearest')  # resize to SwinUNetR input size

        inputs = Variable(inputs.to(device).float())
        label_weight = Variable(label_weight.to(device).float())
        mask = Variable(mask.to(device).float())
        labels = list(labels)
        new_labels_list = []
        for idx, label in enumerate(labels):
            if idx == 0:
                new_labels_list.append(Variable(label.to(device).float()))
            elif idx > 0:
                # set label to -1 outside the brain mask
                label[mask.repeat(1, 5, 1, 1, 1) == 0] = -1
                new_label = Variable(label.to(device))
                new_labels_list.append(new_label)
        predicts = architecture(inputs)
        loss = 0
        if "UncertaintyLoss" in str(type(loss_list[0])) or "TestLoss" in str(type(loss_list[0])):
            loss += loss_list[0](predicts, new_labels_list, label_weight, mask)
        elif len(loss_list) == 1:  # single loss function
            loss += loss_list[0](predicts, new_labels_list[0], label_weight, mask)
        else:  # multiple loss functions, weighted sum
            for idx, (criteria, criteria_weight) in enumerate(zip(loss_list, loss_weights)):
                predict = predicts[idx] if idx < len(predicts) else predicts[len(predicts) - 1]
                if idx > 0:
                    loss = loss.to(criteria.device)
                loss += criteria_weight * criteria(predict, new_labels_list[idx], label_weight, mask)
        optimizer.zero_grad()
        loss.to(device)
        loss.backward()
        optimizer.step()
        training_loss += float(loss.item())
    return training_loss


def calculate_validate_loss(device):
    validation_loss = 0

    # network.eval()
    with torch.no_grad():
        for batch_idx, (inputs, *labels, label_weight, mask, file) in enumerate(validation_loader):
            inputs = F.interpolate(inputs, size=(28, 224, 224), mode='nearest')  # resize to SwinUNetR input size
            label_weight = F.interpolate(label_weight, size=(28, 224, 224), mode='nearest')  # resize to SwinUNetR input size
            mask = F.interpolate(mask, size=(28, 224, 224), mode='nearest')  # resize to SwinUNetR input size

            # print(file)
            inputs = Variable(inputs.to(device))
            labels = list(labels)
            new_labels_list = []
            for idx, label in enumerate(labels):
                if idx == 0:
                    new_labels_list.append(Variable(label.to(device)))
                elif idx > 0:
                    label[mask.repeat(1, 5, 1, 1, 1) == 0] = -1
                    new_label = Variable(label.to(device))
                    new_labels_list.append(new_label)
            label_weight = Variable(label_weight.to(device))
            mask = Variable(mask.to(device))
            predicts = architecture(inputs)
            loss = 0
            if "UncertaintyLoss" in str(type(val_loss_list[0])) or "TestLoss" in str(type(val_loss_list[0])):
                loss += val_loss_list[0](predicts, new_labels_list, label_weight, mask)
            elif len(val_loss_list) == 1:
                if type(predicts) is tuple:
                    loss += val_loss_list[0](predicts[0], new_labels_list[0], label_weight, mask)
                else:
                    loss += val_loss_list[0](predicts, new_labels_list[0], label_weight, mask)
            else:
                for idx, (criteria, criteria_weight) in enumerate(zip(val_loss_list, val_loss_weights)):
                    predict = predicts[idx] if idx < len(predicts) else predicts[len(predicts) - 1]
                    loss += criteria_weight * criteria(predict, new_labels_list[idx], label_weight, mask)
            validation_loss += loss
            loss.to(device)
    return validation_loss


def train_and_evaluate_model(network, device):
    lowest_validation_loss = 1
    best_model_dir = ""
    start = datetime.now()
    epochs = 300
    for epoch in range(epochs):
        network.train()
        try:
            if "StepLR" in str(type(scheduler)):
                scheduler.step()
                print(optimizer.param_groups[0]['lr'])
            elif "Cosine" in str(type(scheduler)):
                scheduler.step()
                print(optimizer.param_groups[0]['lr'])
        except Exception as e:
            print(f"StepLR exception: {e}")
        training_loss = calculate_train_loss(device)
        training_loss /= len(train_loader)
        print(f"[{(datetime.now() - start).total_seconds()}] epoch: {epoch}/{epochs}. Training Loss: {training_loss}")
        wandb_logger.log({"epoch": epoch}, commit=False)
        wandb.log({"epoch": epoch}, step=epoch)
        wandb_logger.log({"Training Loss": training_loss}, commit=False)
        wandb.log({"Training Loss": training_loss}, step=epoch)
        board_writer.add_scalar("training_loss", training_loss, epoch)

        validation_leap = 4
        if epoch % validation_leap == 0 or epoch == epochs:
            network.eval()
            validation_loss = calculate_validate_loss(device)
            validation_loss /= len(validation_loader)
            try:
                if "ReduceLROnPlateau" in str(type(scheduler)):
                    scheduler.step(validation_loss)
                    print(" >> lr :", optimizer.param_groups[0]['lr'])
                    wandb_logger.log({"lr": optimizer.param_groups[0]['lr']}, commit=False)
                    wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=epoch)
            except Exception as e:
                print(f"ReduceLROnPlateau exception: {e}")
            save_model(epoch, validation_loss)
            if lowest_validation_loss == 0 or validation_loss < lowest_validation_loss:
                lowest_validation_loss = validation_loss
                best_model_dir = f"{model_save_dir}/model_{epoch}.pth"
                print(" >> best model :", best_model_dir)
                wandb_logger.log({"best model": best_model_dir}, commit=False)
                wandb.log({"best model": best_model_dir}, step=epoch)
            now = datetime.now()
            print(f" >> [{(now - start).total_seconds()}] Validation Loss: {validation_loss}")
            wandb_logger.log({"Validation Loss": validation_loss}, commit=False)
            wandb.log({"Validation Loss": validation_loss}, step=epoch)
            board_writer.add_scalar("validation_loss", validation_loss, epoch)


if __name__ == '__main__':
    time = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%Hh%Mm')

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='mobilevit')
    parser.add_argument('-u', '--upsampling', type=str, default='deconv')
    parser.add_argument('-d', '--device', type=int, default=4)
    parser.add_argument('-ss', '--slice', type=str, default='False')  # same slice

    # python train_script.py -m 3dmrod -d 4 -ss True
    # python train_script.py -m mobilevit3d -u nearest -d 7 -ss False

    args = vars(parser.parse_args())

    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb_logger = wandb.init(project=f"mra_lightweight",
                              job_type="train")

    model_save_dir = os.path.abspath(f"/data1/sumin/compu/mra_lightweight/{args['model']}_{time}/pretrain")
    summary_writer_folder_dir = os.path.abspath(f"/data1/sumin/compu/mra_lightweight/{args['model']}_{time}/runs")

    board_writer = SummaryWriter(summary_writer_folder_dir)

    # '''model architecture'''
    params = config_mra.network_architecture['parameters']
    architecture = Mobilevit(**params)


    '''dataloader'''
    # train dataset
    dataset_config = config_mra.dataset_train.get('dataset')
    loader_config = config_mra.dataset_train.get('dataloader')

    train_dataset = load_nest_config(**dataset_config)
    train_loader = data.DataLoader(train_dataset, **loader_config)

    # validation dataset
    dataset_config = config_mra.dataset_validation.get('dataset')
    loader_config = config_mra.dataset_validation.get('dataloader')

    validation_dataset = load_nest_config(**dataset_config)
    validation_loader = data.DataLoader(validation_dataset, **loader_config)

    '''loss function'''
    loss_list = []
    loss_weights = [1]
    # config_mra.loss_fn['parameters']['device'] = params['device']
    loss = load_nest_config(**config_mra.loss_fn)
    loss_list.append(loss)
    print("loss function(s):", loss)

    val_loss_list = []
    val_loss_weights = [1]
    # config_mra.val_loss_fn['parameters']['device'] = params['device']
    val_loss = load_nest_config(**config_mra.val_loss_fn)
    val_loss_list.append(val_loss)
    # print("validation loss function(s):", val_loss)

    '''optimizer'''
    optimizer_name = config_mra.optimizer_init['name']
    optimizer_parameters = config_mra.optimizer_init['parameters']
    init_setup = optimizer_parameters['init_setup']
    init_lr = init_setup['lr']

    params = [{'params': architecture.parameters(), 'lr': init_lr},
              {'params': loss_list[0].parameters(), 'lr': init_lr}] \
        if "UncertaintyLoss" in str(type(loss_list[0])) \
        else architecture.parameters()

    optimizer = getattr(optim, optimizer_name)(params, **init_setup)

    '''lr scheduler'''
    lr_scheduler_name = config_mra.learning_rate_scheduler['name']
    lr_scheduler_params = config_mra.learning_rate_scheduler['parameters']
    scheduler = getattr(lr_scheduler, lr_scheduler_name)(optimizer, **lr_scheduler_params)

    print(f"{args['model']}_{args['upsampling']}_{time}")

    '''params'''
    totalparams = 0
    for name, param in architecture.named_parameters():
        if param.requires_grad:
            # print(f"Parameter name: {name}, {param.numel()}")
            totalparams += param.numel()
    print(f">> the total parameters: {totalparams}")

    buffer_size = 0
    for buffer in architecture.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (totalparams + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB\n'.format(size_all_mb))

    device = config_mra.network_architecture['parameters']['device']
    train_and_evaluate_model(architecture, device)

    # img = torch.rand(1, 40, 20, 224, 224).to(device)
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 10
    #
    # import numpy as np
    #
    # timings = np.zeros((repetitions, 1))
    #
    # _ = architecture(img)
    #
    # # MEASURE PERFORMANCE
    # total_time = 0
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter.record()
    #         _ = architecture(img)
    #         ender.record()
    #         # WAIT FOR GPU SYNC
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[rep] = curr_time
    #         total_time += curr_time
    #
    # print(total_time / repetitions)
