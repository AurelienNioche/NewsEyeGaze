import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(4))

    def draw_samples(self, n_sample=100, compute_penalty=False, lbd_scale=2, k_scale=2):

        unc_k_mu, unc_k_logvar, unc_lbd_mu, unc_lbd_logvar = self.param

        unc_k_sd = (0.5 * unc_k_logvar).exp()
        unc_lbd_sd = (0.5 * unc_lbd_logvar).exp()

        unc_k = torch.randn(size=(n_sample, 1)) * unc_k_sd + unc_k_mu
        unc_lbd = torch.randn(size=(n_sample, 1)) * unc_lbd_sd + unc_lbd_mu

        k = torch.sigmoid(unc_k) * k_scale
        lbd = torch.sigmoid(unc_lbd) * lbd_scale

        if compute_penalty:
            pen_unc_k = torch.distributions.Normal(unc_k_mu, unc_k_sd).log_prob(unc_k)
            pen_unc_lbd = torch.distributions.Normal(unc_lbd_mu, unc_lbd_sd).log_prob(unc_lbd)

            penalty = pen_unc_k + pen_unc_lbd

            return k, lbd, penalty

        else:
            return k, lbd

    def forward(self, i, d, y, n_sample=100):

        k, lbd, penalty = self.draw_samples(compute_penalty=True, n_sample=n_sample)

        lls = torch.distributions.Weibull(k, lbd).log_prob(y).sum(axis=-1)
        lls_mean = lls.mean()
        to_min = - lls_mean + penalty.mean()
        return to_min


class Dataset(torch.utils.data.Dataset):

    def __init__(self, condition):
        self.i, self.d, self.y = self.load_data(condition)

    def load_data(self, condition):
        df = pd.read_csv("data/stan_data_covariates.csv", index_col=0)
        df = df[df.x == condition]
        i = torch.tensor(df.i.values, dtype=torch.float)
        d = torch.tensor(df.d.values, dtype=torch.float)
        y = torch.tensor(df.y.values, dtype=torch.float)

        # y = torch.distributions.Weibull()

        return i, d, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.i[idx], self.d[idx], self.y[idx]


def main():

    condition = 1

    torch.manual_seed(123)

    data = Dataset(condition=condition)

    n_obs = len(data)
    print("n observation", n_obs)

    model = Model()

    dataloader = torch.utils.data.DataLoader(data, batch_size=len(data))

    optimizer = torch.optim.Adam(lr=0.1, params=model.parameters())

    n_epochs = 300
    n_sample = 10
    hist_loss = []

    for _ in tqdm(range(n_epochs)):
        for batch, (i, d, y) in enumerate(dataloader):
            loss = model(i=i, d=d, y=y, n_sample=n_sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hist_loss.append(loss.item())

    fig, ax = plt.subplots()
    ax.set_title("loss")
    ax.plot(hist_loss)
    plt.show()

    with torch.no_grad():
        k, lbd = model.draw_samples(n_sample=10000)

    print(f"k={k.mean()}, lbd={lbd.mean()}")

    fig, ax = plt.subplots()
    sns.histplot(k.detach().numpy(), ax=ax, stat='density', legend=False)
    ax.set_title("k")
    plt.show()

    fig, ax = plt.subplots()
    sns.histplot(lbd.detach().numpy(), ax=ax, stat='density', legend=False)
    ax.set_title("lbd")
    plt.show()


if __name__ == "__main__":
    main()
