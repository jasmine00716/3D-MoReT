from writer.writer import Writer
import torch


class DscMrpCsvWriter(Writer):
    def write(self, config):
        if config["output_file"] is None:
            if self.output_file is None:
                raise AttributeError
            else:
                file = self.output_file
        else:
            file = config["output_file"]
        if config["data"] is None:
            raise ValueError
        with open(file, 'w') as file:
            for k, v in config["data"].items():
                v = torch.FloatTensor(v)
                file.writelines(f"{k}.mean: {v.mean(0)}.\n")
                file.writelines(f"{k}.median: {v.median(0)[0]}.\n")
                file.writelines(f"{k}.SD: {v.std(0)}.\n")
                # file.writelines(f"{k}.min: {v.min(0)[0]}.\n")
                # file.writelines(f"{k}.max: {v.max(0)[0]}.\n")
                file.writelines(f"{k}.min_index: {v.min(0)[1]}.\n")
                file.writelines(f"{k}.max_index: {v.max(0)[1]}.\n")
