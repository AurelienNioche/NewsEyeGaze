import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(2))

    def forward(self, y):
        mu, logvar = self.param
        sd = (0.5*logvar).exp()
        lls = torch.distributions.Normal(mu, sd).log_prob(y).sum()
        return - lls


class Dataset(torch.utils.data.Dataset):

    def __init__(self):
        self.y = self.generate_data()

    @staticmethod
    def generate_data():
        n = 4000
        torch.manual_seed(123)
        y = torch.distributions.Normal(2.0, 3.0).sample((n,))
        return y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.y[idx]


def main():

    data = Dataset()

    n_obs = len(data)
    print("n observation", n_obs)

    model = Model()

    dataloader = torch.utils.data.DataLoader(data, batch_size=len(data))

    optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())

    n_epochs = 300
    hist_loss = []

    for _ in tqdm(range(n_epochs)):
        for batch, y in enumerate(dataloader):
            loss = model(y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hist_loss.append(loss.item())

    fig, ax = plt.subplots()
    ax.set_title("loss")
    ax.plot(hist_loss)
    plt.show()

    print(f"mu = {model.param[0].item():.2f} "
          f"sd = {(0.5*model.param[1]).exp().item():.2f}")


if __name__ == "__main__":
    main()
