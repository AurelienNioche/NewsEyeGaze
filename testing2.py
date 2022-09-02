import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(4))

    def forward(self, y, n_sample=5000):
        mu_mu, mu_logvar, logvar_mu, logvar_logvar = self.param

        mu_sd = (0.5 * mu_logvar).exp()
        logvar_sd = (0.5 * logvar_logvar).exp()

        mu = torch.randn(size=(n_sample, 1)) * mu_sd + mu_mu
        logvar = torch.randn(size=(n_sample, 1)) * logvar_sd + logvar_mu

        pen_mu = torch.distributions.Normal(mu_mu, mu_sd).log_prob(mu)
        pen_logvar = torch.distributions.Normal(logvar_mu, logvar_sd).log_prob(logvar)

        penalty = pen_mu + pen_logvar

        sd = (0.5*logvar).exp()
        lls = torch.distributions.Normal(mu, sd).log_prob(y).sum(axis=-1)
        return - lls.mean() + penalty.mean()


class Dataset(torch.utils.data.Dataset):

    def __init__(self):
        self.y = self.generate_data()

    @staticmethod
    def generate_data():
        n = 10000
        # k = 0.90
        # lbda = 1.04
        # y = torch.distributions.Weibull(concentration=k, scale=lbda).sample((n,))
        y = torch.distributions.Normal(2.0, 2.0).sample((n,))
        return y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.y[idx]


def main():
    torch.manual_seed(123)

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

    print(f"MU mu = {model.param[0].item():.2f} "
          f"sd = {(0.5*model.param[1]).exp().item():.2f}")

    print(f"SIGMA mu = {model.param[2].item():.2f} "
          f"sd = {(0.5*model.param[3]).exp().item():.2f}")


if __name__ == "__main__":
    main()
