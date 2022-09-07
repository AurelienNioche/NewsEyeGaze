import os

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(8))

    def draw_samples(self, i, d, n_sample=100, compute_penalty=False, lbd_scale=2, k_scale=2):

        alpha_mu, alpha_logvar, \
            beta0_mu, beta0_logvar, \
            betai_mu, betai_logvar, \
            betad_mu, betad_logvar \
            = self.param

        alpha_sd = (0.5 * alpha_logvar).exp()
        beta0_sd = (0.5 * beta0_logvar).exp()
        betai_sd = (0.5 * betai_logvar).exp()
        betad_sd = (0.5 * betad_logvar).exp()

        alpha = torch.randn(size=(n_sample, 1)) * alpha_sd + alpha_mu

        beta_0 = torch.randn(size=(n_sample, 1)) * beta0_sd + beta0_mu
        beta_i = torch.randn(size=(n_sample, 1)) * betai_sd + betai_mu
        beta_d = torch.randn(size=(n_sample, 1)) * betad_sd + betad_mu

        unc_k = alpha
        unc_lbd = -(beta_0 + beta_i * i + beta_d * d)  # sigma in Stan, lambda/scale in Wikipdedia, scale in Torch

        k = torch.sigmoid(unc_k) * k_scale        # alpha in Stan, k/shape in Wikipedia, k/contentration in Torch
        lbd = torch.sigmoid(unc_lbd) * lbd_scale  # sigma in Stan, lambda/scale in Wikipdedia, scale in Torch

        smp = {"k": k, "lbd": lbd,
               "beta_0": beta_0, "beta_i": beta_i, "beta_d": beta_d}
        if compute_penalty:

            penalty = 0
            for (mu, sd, samples) in ((alpha_mu, alpha_sd, alpha),
                                      (beta0_mu, beta0_sd, beta_0),
                                      (betai_mu, betai_sd, beta_i),
                                      (betad_mu, betad_sd, beta_d)):

                penalty += torch.distributions.Normal(mu, sd).log_prob(samples)

            return smp, penalty

        return smp

    def forward(self, i, d, y, n_sample=100):

        smp, penalty = \
            self.draw_samples(compute_penalty=True,
                              n_sample=n_sample,
                              i=i, d=d)

        lls = torch.distributions.Weibull(smp['k'], smp['lbd']).log_prob(y).sum(axis=-1)
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

    fig_folder = "fig"
    os.makedirs(fig_folder, exist_ok=True)

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
    plt.savefig(f"{fig_folder}/loss.pdf")
    plt.show()

    with torch.no_grad():
        smp = model.draw_samples(n_sample=10000, i=data.i, d=data.d)

    for key, v in smp.items():
        if key == "lbd":
            v = v.mean(axis=-1)

        print(f"{key}={v.mean():.3f}")

        fig, ax = plt.subplots()
        sns.histplot(v.detach().numpy(), ax=ax, stat='density', legend=False)
        ax.set_title(key)
        plt.savefig(f"{fig_folder}/{key}_samples.pdf")
        plt.show()


if __name__ == "__main__":
    main()
