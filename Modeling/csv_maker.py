import csv
import glob
import pandas as pd

header = ["approach", "model", "dataset", "name", "MAE", "MAPE", "RMSE", "MASE", "train time", "inf time", "model size"]
with open("models_csv.csv", 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

starting_path = "./logs"
if __name__ == "__main__":
    with open("models_csv.csv", 'a', encoding='UTF8') as f:
        files = glob.glob('./logs/**/**/**/**/metrics.txt', recursive=True)
        writer = csv.writer(f)
        for file in files:
            data = pd.read_csv(file, sep=',', index_col=0, engine='python')
            to_write = [file.split('/')[2], file.split('/')[3], file.split('/')[4], file.split('/')[5],
                        data.iloc[0, 0], data.iloc[1, 0], data.iloc[2, 0], data.iloc[3, 0], data.iloc[4, 0],
                        data.iloc[5, 0], data.iloc[6, 0]]
            writer.writerow(to_write)
