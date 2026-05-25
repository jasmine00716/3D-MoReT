class BaseConfig:
    def __init__(self):
        self.epochs = 300
        self.pretrain = False
        self.train_summary_writer_folder = "training_loss"
        self.val_summary_writer_folder = "validation_loss"
        self.validation_leap = 2
        self.model_parallel = False
        self.network_architecture = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.loss = None
        self.loss_weights = None
        self.val_loss = None
        self.val_loss_weights = None
        self.optimizer = None
        self.lr_scheduler = None
        self.summary_writer_folder_dir = "/data1/sumin/compu/3d_mrod/runs"
        self.log_file_dir = "/data1/sumin/compu/3d_mrod/log_file"
        self.model_save_dir = "/data1/sumin/compu/3d_mrod/pretrain"
        self.predict_save_dir = "/data1/sumin/compu/3d_mrod/result"
        self.evaluation_metrics_save_dir = "/data1/sumin/compu/3d_mrod/evaluation_metrics"

if __name__ == '__main__':
    base_config = BaseConfig()
    print(base_config.__dict__)