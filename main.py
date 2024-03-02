import torch

if __name__ == "__main__":
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(f"{x_data=}")
