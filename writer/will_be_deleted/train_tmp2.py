'''
deconvolution 연산 대신 nearest neighbor upsampling으로 대체한 모델

train, validation에 같은 loss function 적용

'''




import models.MobileViT.mobilevit
import os
import torch
from datetime import datetime
from pytz import timezone
from utils_ import load_nest_config
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import will_be_deleted.config_mobilevit as config
import wandb


def calculate_train_loss():
    training_loss = 0
    device = torch.device("cuda:6")

    for batch_idx, (inputs, *labels, label_weight, mask, file) in enumerate(train_loader):
        # print(inputs.shape)

        inputs = Variable(inputs.to(device))
        label_weight = Variable(label_weight.to(device))
        mask = Variable(mask.to(device))
        labels = list(labels)
        new_labels_list = []
        for idx, label in enumerate(labels):
            if idx == 0:
                new_labels_list.append(Variable(label.to(device)))
            elif idx > 0:
                # mask 배열의 원소가 0인지 아닌지를 나타내는 boolean 배열 생성
                # True인 label 배열의 요소들을 선택하고 -1로 값을 바꿔줌
                label[mask.repeat(1, 5, 1, 1, 1) == 0] = -1
                new_label = Variable(label.to(device))
                new_labels_list.append(new_label)
        predicts = network(inputs)
        loss = 0
        if len(loss_list) == 1:  # loss function 1개만 사용할 경우
            loss += loss_list[0](predicts, new_labels_list[0], label_weight, mask)
        else:  # loss function 2개이상 사용할 경우(regression + ordinal regression 등등)
            for idx, (criteria, criteria_weight) in enumerate(zip(loss_list, loss_weights)):
                predict = predicts[idx] if idx < len(predicts) else predicts[len(predicts) - 1]
                loss += criteria_weight * criteria(predict, new_labels_list[idx], label_weight, mask)
        optimizer.zero_grad()
        loss.to(device)
        loss.backward()
        optimizer.step()
        training_loss += float(loss.item())
    return training_loss

def save_model(epoch, validation_loss):
    loss_state_dict = val_loss_list[0].state_dict() if "UncertaintyLoss" in str(type(val_loss_list[0])) else None

    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir, exist_ok = True)

    torch.save({
            "epoch": epoch,
            "model_state_dict": network.state_dict(),
            "loss_state_dict": loss_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": validation_loss,
            "scheduler": scheduler.state_dict(),
        }, f"{model_save_dir}/model_{epoch}.pth"
    )


def calculate_validate_loss():
    validation_loss = 0
    device = torch.device("cuda:6")

    # network.eval()
    with torch.no_grad():
        for index, (inputs, labels, label_weight, mask, file) in enumerate(validation_loader):
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
            predicts = network(inputs)
            loss = 0
            if len(val_loss_list) == 1:
                if type(predicts) is tuple:
                    loss += val_loss_list[0](predicts[0], new_labels_list[0], label_weight, mask)
                else:
                    loss += val_loss_list[0](predicts, new_labels_list[0], label_weight, mask)
            else:
                for idx, (criteria, criteria_weight) in enumerate(zip(val_loss_list, val_loss_weights)):
                    predict = predicts[idx] if idx < len(predicts) else predicts[len(predicts) - 1]
                    loss += criteria_weight * criteria(predict, new_labels_list[idx], label_weight, mask)
            validation_loss += float(loss)
            loss.to(device)
    return validation_loss

def train():
    lowest_validation_loss = 1
    best_model_dir = ""
    start = datetime.now()
    epochs = 300
    for epoch in range(epochs): # 10으로 test
        network.train()
        try:
            if "StepLR" in str(type(scheduler)):
                scheduler.step()
                print(optimizer.param_groups[0]['lr'])
        except Exception as e:
            print(f"StepLR exception: {e}")

        training_loss = calculate_train_loss()
        training_loss /= len(train_loader)
        print(f"[{(datetime.now() - start).total_seconds()}] epoch: {epoch}/{epochs}. Training Loss: {training_loss}")
        wandb_logger.log({"epoch": epoch}, commit=False)
        wandb.log({"epoch": epoch}, step=epoch)
        wandb_logger.log({"Training Loss": training_loss}, commit=False)
        wandb.log({"Training Loss": training_loss}, step=epoch)
        board_writer.add_scalar("training_loss", training_loss, epoch)

        if epoch % validation_leap == 0 or epoch == epochs:
            network.eval()
            validation_loss = calculate_validate_loss()
            validation_loss /= len(validation_loader)
            try:
                if "ReduceLROnPlateau" in str(type(scheduler)):
                    scheduler.step(validation_loss)
                    print(">> lr :", optimizer.param_groups[0]['lr'])
                    wandb_logger.log({"lr": optimizer.param_groups[0]['lr']}, commit=False)
                    wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=epoch)
            except Exception as e:
                print(f"ReduceLROnPlateau exception: {e}")
            save_model(epoch, validation_loss)
            if lowest_validation_loss == 0 or validation_loss < lowest_validation_loss:
                lowest_validation_loss = validation_loss
                best_model_dir = f"{model_save_dir}/model_{epoch}.pth"
                print(">> best model :", best_model_dir)
                wandb_logger.log({"best model": best_model_dir}, commit=False)
                wandb.log({"best model": best_model_dir}, step=epoch)
            now = datetime.now()
            print(f">> [{(now - start).total_seconds()}] Validation Loss: {validation_loss}")
            wandb_logger.log({"Validation Loss": validation_loss}, commit=False)
            wandb.log({"Validation Loss": validation_loss}, step=epoch)
            board_writer.add_scalar("validation_loss", validation_loss, epoch)


if __name__ == '__main__':
    seed = 42
    cuda_visible_devices = [6, 7]

    time = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%Hh%Mm')

    log_file_dir = os.path.abspath(f"/data1/sumin/compu/mrp_lightweight_{time}/log_file")
    model_save_dir = os.path.abspath(f"/data1/sumin/compu/mrp_lightweight_{time}/pretrain")
    summary_writer_folder_dir = os.path.abspath(f"/data1/sumin/compu/mrp_lightweight_{time}/runs")

    board_writer = SummaryWriter(summary_writer_folder_dir)

    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb_logger = wandb.init(project=f"mrp_lightweight_model",
                              job_type="train",
                              name=f"{log_file_dir.split('/')[-1]}")

    validation_leap = 4

    # # 분산 데이터 병렬화 설정
    # dist.init_process_group(backend='nccl', init_method='env://127.0.0.1:40', world_size=1, rank=0)

    # model_parallel = False
    params = config.network_architecture['parameters']
    params['dev_0'] = torch.device("cuda:6")
    params['dev_1'] = torch.device("cuda:6")
    params['dev_2'] = torch.device("cuda:6")
    params['dev_3'] = torch.device("cuda:6")
    network = models.MobileViT_3D.mobilevit.Mobilevit_unet_nearest_neighbor_upsampling(**params)

    totalparams = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"the total number of parameters: {totalparams}")

    buffer_size = 0
    for buffer in network.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (totalparams + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB\n'.format(size_all_mb))

    loss_list = []
    loss_weights = [1]
    config.loss_fn['parameters']['device'] = torch.device("cuda:6")
    loss = load_nest_config(**config.loss_fn)
    loss_list.append(loss)
    print("loss function(s):", loss)

    val_loss_list = []
    val_loss_weights = [1]

    config.val_loss_fn['parameters']['device'] = torch.device("cuda:6")
    val_loss = load_nest_config(**config.val_loss_fn)
    val_loss_list.append(val_loss)
    print("validation loss function(s):", val_loss)

    optimizer_name = config.optimizer_init['name']
    optimizer_parameters = config.optimizer_init['parameters']
    init_setup = optimizer_parameters['init_setup']
    init_lr = init_setup['lr']

    params = [{'params': network.parameters(), 'lr': init_lr},
              {'params': loss_list[0].parameters(), 'lr': init_lr}] \
        if "UncertaintyLoss" in str(type(loss_list[0])) \
        else network.parameters()

    optimizer = getattr(optim, optimizer_name)(params, **init_setup)

    lr_scheduler_name = config.learning_rate_scheduler['name']
    lr_scheduler_params = config.learning_rate_scheduler['parameters']
    scheduler = getattr(lr_scheduler, lr_scheduler_name)(optimizer, **lr_scheduler_params)

    # train dataset
    dataset_config = config.dataset_train.get('dataset')
    loader_config = config.dataset_train.get('dataloader')

    train_dataset = load_nest_config(**dataset_config)
    train_loader = data.DataLoader(train_dataset, **loader_config)

    # validation dataset
    dataset_config = config.dataset_validation.get('dataset')
    loader_config = config.dataset_validation.get('dataloader')

    validation_dataset = load_nest_config(**dataset_config)
    validation_loader = data.DataLoader(validation_dataset, **loader_config)

    # world_size = 2
    # mp.spawn(train, args=(world_size,), nprocs=world_size)
    train()
