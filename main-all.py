import os

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Dataset(torch.utils.data.Dataset):

    def __init__(self, condition):
        self.condition = condition

        self.x_loc, self.y_loc, self.i, self.d, self.y = self.load_data(condition)
        self.covar = "x_loc", "y_loc", "i", "d"
        if condition == 1:
            self.covar = self.covar[1:]
        self.n_covar = len(self.covar)

    def load_data(self, condition):
        df = pd.read_csv("data/stan_data_all.csv", index_col=0)
        df = df[df.x == condition]
        x_loc = torch.tensor(df.xloc.values, dtype=torch.float)
        y_loc = torch.tensor(df.yloc.values, dtype=torch.float)
        x_loc -= 0.5
        y_loc -= 0.5
        i = torch.tensor(df.i.values, dtype=torch.float)
        d = torch.tensor(df.d.values, dtype=torch.float)

        y = torch.tensor(df.y.values, dtype=torch.float)
        return x_loc, y_loc, i, d, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        covar = [self.x_loc[idx], self.y_loc[idx], self.i[idx], self.d[idx]]
        if self.condition == 1:
            covar = covar[1:]
        return covar, self.y[idx]


class Model(torch.nn.Module):
    def __init__(self, n_covar):
        super().__init__()
        self.n_covar = n_covar

        self.logvar = torch.nn.Parameter(torch.zeros(n_covar))
        self.mu = torch.nn.Parameter(torch.zeros(n_covar))

        self.alpha_param = torch.nn.Parameter(torch.zeros(2))

    def beta_dist_parameters(self):

        param = []

        for i in range(self.n_covar):
            logvar = self.logvar[i]
            mu = self.mu[i]
            sd = (0.5 * logvar).exp()
            param.append((mu, sd))

        return param

    def forward(self, covar, y, n_sample=100, compute_penalty=False,
                lbd_scale=2, k_scale=2):  # !!! Be aware of this arbitrary choice

        alpha_mu, alpha_logvar = self.alpha_param
        alpha_sd = (0.5 * alpha_logvar).exp()
        alpha_smp = torch.randn(size=(n_sample, 1)) * alpha_sd + alpha_mu
        unc_k = alpha_smp

        penalty = torch.distributions.Normal(alpha_mu, alpha_sd).log_prob(alpha_smp)

        random_n = torch.randn(size=(self.n_covar, n_sample, 1))

        sum_beta = torch.zeros((n_sample, len(y)))
        for i in range(self.n_covar):
            logvar = self.logvar[i]
            mu = self.mu[i]
            rd = random_n[i]
            sd = (0.5 * logvar).exp()
            beta_smp = rd * sd + mu
            if i > 0:
                sum_beta += beta_smp*covar[i]
            else:
                sum_beta += beta_smp

            penalty += torch.distributions.Normal(mu, sd).log_prob(beta_smp)

        unc_lbd = -sum_beta  # sigma in Stan, lambda/scale in Wikipdedia, scale in Torch
        # !!! Not divided by k, contrary to the article

        k_smp = 1e-7 + torch.sigmoid(unc_k) * k_scale        # alpha in Stan, k/shape in Wikipedia, k/contentration in Torch
        lbd_smp = 1e-7 + torch.sigmoid(unc_lbd) * lbd_scale  # sigma in Stan, lambda/scale in Wikipdedia, scale in Torch

        lls = torch.distributions.Weibull(k_smp, lbd_smp).log_prob(y).sum(axis=-1)
        lls_mean = lls.mean()
        to_min = - lls_mean + penalty.mean()
        return to_min


def main():

    condition = 2  # 1: Single column / 2: double-column

    n_epochs = 300
    n_sample = 40
    lbd_scale = 2
    k_scale = 2
    learning_rate = 0.05

    seed = 1234

    fig_folder = "fig/main-all"
    os.makedirs(fig_folder, exist_ok=True)

    torch.manual_seed(seed)

    data = Dataset(condition=condition)

    n_obs = len(data)
    print("n observation", n_obs)

    model = Model(data.n_covar)

    dataloader = torch.utils.data.DataLoader(data, batch_size=len(data))

    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())

    hist_loss = []

    for _ in tqdm(range(n_epochs)):
        for batch, (covar, y) in enumerate(dataloader):
            loss = model(covar=covar, y=y, n_sample=n_sample,
                         lbd_scale=lbd_scale, k_scale=k_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hist_loss.append(loss.item())

    fig, ax = plt.subplots()
    ax.set_title("loss")
    ax.plot(hist_loss)
    plt.savefig(f"{fig_folder}/loss.pdf")
    plt.show()

    # with torch.no_grad():
    #     smp = model.draw_samples(n_sample=10000, x_loc=data.x_loc, y_loc=data.y_loc)
    #
    # for key, v in smp.items():
    #     if key == "lbd":
    #         v = v.mean(axis=-1)
    #     elif key.startswith("beta"):
    #         continue
    #
    #     m = v.mean()
    #     print(f"{key}={m:.3f}")
    #
    #     fig, ax = plt.subplots()
    #     sns.histplot(v.detach().numpy(), ax=ax, stat='density', legend=False)
    #     ax.set_title(key)
    #     plt.savefig(f"{fig_folder}/{key}_samples.pdf")
    #     plt.show()

    with torch.no_grad():
        beta_dist_prm = model.beta_dist_parameters()

        for i in range(data.n_covar):

            key = data.covar[i]
            mu, sd = beta_dist_prm[i]

            fig, ax = plt.subplots()

            lower, upper = mu-1.96*sd, mu+1.96*sd
            print(f"{key}={mu:.3f} [{lower:.3f}, {upper:.3f}]")

            x = torch.linspace(mu-3*sd, mu+3*sd, 100)
            y = torch.distributions.Normal(mu, sd).log_prob(x).exp()
            x = x.numpy()
            y = y.numpy()

            sns.lineplot(x=x, y=y, ax=ax)
            ax.set_title(key)
            for bound in lower, upper:
                ax.vlines(x=bound, ymin=0,
                          ymax=torch.distributions.Normal(mu, sd).log_prob(torch.tensor([bound, ])).exp().numpy(),
                          color="red", ls="--")
            plt.savefig(f"{fig_folder}/{key}_dist.pdf")
            plt.show()


if __name__ == "__main__":
    main()
