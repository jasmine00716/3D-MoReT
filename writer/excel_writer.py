from writer.writer import Writer
import torch
import pandas as pd


class DscMrpExcelWriter(Writer):
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
        em = ["R-Squared", "MAE", "TM", "SSIM"]
        cols = pd.MultiIndex.from_product([em, ["Art", "Cap", "EVen", "LVen", "Del"]])
        new_df = pd.DataFrame(columns=cols)
        for k, v in config["data"].items():
            v = torch.FloatTensor(v)
            df = pd.DataFrame(data=v[:, :], index=[i for i in range(v.shape[0])], columns=[j for j in range(v.shape[1])])
            mean = df.mean()
            median = df.median()
            std = df.std()
            minn = df.min()
            maxx = df.max()
            new_df.at["Mean", (k, "Art")] = mean[0]
            new_df.at["Mean", (k, "Cap")] = mean[1]
            new_df.at["Mean", (k, "EVen")] = mean[2]
            new_df.at["Mean", (k, "LVen")] = mean[3]
            new_df.at["Mean", (k, "Del")] = mean[4]
            new_df.at["Median", (k, "Art")] = median[0]
            new_df.at["Median", (k, "Cap")] = median[1]
            new_df.at["Median", (k, "EVen")] = median[2]
            new_df.at["Median", (k, "LVen")] = median[3]
            new_df.at["Median", (k, "Del")] = median[4]
            new_df.at["SD", (k, "Art")] = std[0]
            new_df.at["SD", (k, "Cap")] = std[1]
            new_df.at["SD", (k, "EVen")] = std[2]
            new_df.at["SD", (k, "LVen")] = std[3]
            new_df.at["SD", (k, "Del")] = std[4]
            new_df.at["Min", (k, "Art")] = minn[0]
            new_df.at["Min", (k, "Cap")] = minn[1]
            new_df.at["Min", (k, "EVen")] = minn[2]
            new_df.at["Min", (k, "LVen")] = minn[3]
            new_df.at["Min", (k, "Del")] = minn[4]
            new_df.at["Max", (k, "Art")] = maxx[0]
            new_df.at["Max", (k, "Cap")] = maxx[1]
            new_df.at["Max", (k, "EVen")] = maxx[2]
            new_df.at["Max", (k, "LVen")] = maxx[3]
            new_df.at["Max", (k, "Del")] = maxx[4]
        writer = pd.ExcelWriter(file, engine='xlsxwriter')
        new_df.to_excel(writer, sheet_name="evaluation_metrics", index=True)
        writer.save()

class DscMrpAllExcelWriter(Writer):
    def write(self, config, folders):
        if config["output_file"] is None:
            if self.output_file is None:
                raise AttributeError
            else:
                file = self.output_file
        else:
            file = config["output_file"]
        if config["data"] is None:
            raise ValueError
        # em = ["R-Squared", "MAE", "TM", "SSIM"]

        for k, v in config["data"].items():
            em = [f"{k}"]
            cols = pd.MultiIndex.from_product([em, ["Art", "Cap", "EVen", "LVen", "Del"]])
            new_df = pd.DataFrame(columns=cols)
            v = torch.FloatTensor(v)
            df = pd.DataFrame(data=v[:, :], index=[i for i in range(v.shape[0])], columns=[j for j in range(v.shape[1])])
            df["folders"] = folders
            for row in df.iterrows():
                new_df.at[row[1]["folders"], (k, "Art")] = float(row[1][0])
                new_df.at[row[1]["folders"], (k, "Cap")] = float(row[1][1])
                new_df.at[row[1]["folders"], (k, "EVen")] = float(row[1][2])
                new_df.at[row[1]["folders"], (k, "LVen")] = float(row[1][3])
                new_df.at[row[1]["folders"], (k, "Del")] = float(row[1][4])
            writer = pd.ExcelWriter(f"{file}_{k}.xlsx", engine='xlsxwriter')
            new_df.to_excel(writer, sheet_name="evaluation_metrics", index=True)
            writer.save()

