import os
import torch
from datetime import datetime
from utils_ import import_file, convert_str_from_underscore_to_camel
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from evaluation_metrics import Metrics_Evaluation
import will_be_deleted.config_mobilevit as config


def load_nest_config(**kwargs):
    file_name = kwargs.get('file')
    parameters = kwargs.get('parameters')
    if "/" in file_name:
        cls_name = convert_str_from_underscore_to_camel(file_name.split("/")[-1])
    else:
        cls_name = convert_str_from_underscore_to_camel(file_name)
    if parameters is None:
        return getattr(import_file(file_name), cls_name)()
    return getattr(import_file(file_name), cls_name)(**parameters)

def predict():
    import models.MobileViT.mobilevit
    params = config.network_architecture['parameters']


    network = models.MobileViT_3D.mobilevit.Mobilevit_unet_nearest_neighbor_upsampling(**params)
    params = config.network_architecture['parameters']
    params['dev_0'] = torch.device("cuda:7")
    params['dev_1'] = torch.device("cuda:7")
    params['dev_2'] = torch.device("cuda:7")
    params['dev_3'] = torch.device("cuda:7")
    print(config.network_architecture['file'])

    best_model_dir = "/data1/sumin/compu/mrp_lightweight_20231016_00h50m/pretrain/model_172.pth"
    checkpoint = torch.load(best_model_dir, map_location=device)
    network.load_state_dict(checkpoint['model_state_dict'])

    start = datetime.now()
    for index, (inputs, labels, mask, filename) in enumerate(test_loader):
        folder = '/'.join(filename[0].split('/')[-7:-3])
        inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)

        network.eval()
        predicts = network(inputs)
        print(f"[{(datetime.now() - start).total_seconds()}] {index + 1}: {folder}")
        del inputs

        directory = f'{predict_save_dir}/{folder}'
        if not os.path.isdir(directory):
            os.makedirs(directory)

        np.save(f'{predict_save_dir}/{folder}/gt.npy',torch.mul(labels, mask.repeat(1, 5, 1, 1, 1))[0].cpu().detach().numpy())
        del labels
        np.save(f'{predict_save_dir}/{folder}/predict.npy', torch.mul(predicts[0], mask.repeat(1, 5, 1, 1, 1))[0].cpu().detach().numpy())
        del predicts
        np.save(f'{predict_save_dir}/{folder}/mask.npy', mask[0].cpu().detach().numpy())
        del mask
    print(f"\n\n>> total time: {(datetime.now() - start).total_seconds()}")
    print(f">> per patient(mean): {(datetime.now() - start).total_seconds() / len(test_loader)}")


def evaluate(log_file_dir, predict_save_dir, evaluation_metrics_save_dir):
    evaluate_metrics = Metrics_Evaluation(log_file_dir, predict_save_dir, evaluation_metrics_save_dir, save_fig=True)
    evaluate_metrics.evaluate()


if __name__ == '__main__':
    cuda_visible_devices = [4, 5, 6, 7]

    # time = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%Hh%Mm')

    # log_file_dir = os.path.abspath(f"/data1/sumin/compu/mrp_lightweight_{time}/log_file")
    # model_save_dir = os.path.abspath(f"/data1/sumin/compu/mrp_lightweight_{time}/pretrain")
    # predict_save_dir = os.path.abspath(f"/data1/sumin/compu/mrp_lightweight_{time}/result")
    # evaluation_metrics_save_dir = os.path.abspath(f"/data1/sumin/compu/mrp_lightweight_{time}/evaluation_metrics")
    # summary_writer_folder_dir = os.path.abspath(f"/data1/sumin/compu/mrp_lightweight_{time}/runs")
    log_file_dir = os.path.abspath(f"/data1/sumin/compu/mrp_lightweight_20231016_00h50m/log_file")
    model_save_dir = os.path.abspath(f"/data1/sumin/compu/mrp_lightweight_20231016_00h50m/pretrain")
    predict_save_dir = os.path.abspath(f"/data1/sumin/compu/mrp_lightweight_20231016_00h50m/result")
    evaluation_metrics_save_dir = os.path.abspath(f"/data1/sumin/compu/mrp_lightweight_20231016_00h50m/evaluation_metrics")
    summary_writer_folder_dir = os.path.abspath(f"/data1/sumin/compu/mrp_lightweight_20231016_00h50m/runs")

    board_writer = SummaryWriter(summary_writer_folder_dir)

    validation_leap = 4

    # test dataset
    dataset_config = config.dataset_test.get('dataset')
    loader_config = config.dataset_test.get('dataloader')

    test_dataset = load_nest_config(**dataset_config)
    test_loader = data.DataLoader(test_dataset, **loader_config)

    device = torch.device("cuda:7")

    predict()
    evaluate(predict_save_dir, predict_save_dir, evaluation_metrics_save_dir)