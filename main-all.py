import os

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Dataset(torch.utils.data.Dataset):

    def __init__(self, condition):
        self.condition = condition

        self.covar = ["i", "d"]
        if condition == 1 and "xloc" in self.covar:
            self.covar.remove("xloc")

        self.n_covar = len(self.covar)

        self.data = self.load_data(condition)
        self.n_observation = len(self.data["y"])

    def load_data(self, condition):
        df = pd.read_csv("data/stan_data_all.csv", index_col=0)
        df = df[df.x == condition]

        data = {}
        for covar in self.covar:
            values = torch.tensor(df[covar].values, dtype=torch.float)
            if covar in ("xloc", "yloc"):
                values -= 0.5
            data[covar] = values

        data["y"] = torch.tensor(df.y.values, dtype=torch.float)
        return data

    def __len__(self):
        return self.n_observation

    def __getitem__(self, idx):
        covar_data = []
        for covar in self.covar:
            covar_data.append(self.data[covar][idx])
        return covar_data, self.data['y'][idx]


class Model(torch.nn.Module):
    def __init__(self, n_covar, lbd_scale, k_scale):
        super().__init__()
        self.n_covar = n_covar

        self.logvar = torch.nn.Parameter(torch.zeros(n_covar + 1))
        self.mu = torch.nn.Parameter(torch.zeros(n_covar + 1))

        self.alpha_param = torch.nn.Parameter(torch.zeros(2))

        self.lbd_scale = lbd_scale
        self.k_scale = k_scale

    def sample_concentration_parameter(self, n_sample):

        alpha_mu, alpha_logvar = self.alpha_param
        alpha_sd = (0.5 * alpha_logvar).exp()
        alpha_smp = torch.randn(size=(n_sample, 1)) * alpha_sd + alpha_mu
        k_smp = 1e-7 + torch.sigmoid(alpha_smp) * self.k_scale
        return k_smp

    def beta_dist_parameters(self):

        param = []

        for i in range(self.n_covar + 1):
            logvar = self.logvar[i]
            mu = self.mu[i]
            sd = (0.5 * logvar).exp()
            param.append((mu, sd))

        return param

    def forward(self, covar, y, n_sample):  # !!! Be aware of this arbitrary choice

        alpha_mu, alpha_logvar = self.alpha_param
        alpha_sd = (0.5 * alpha_logvar).exp()
        alpha_smp = torch.randn(size=(n_sample, 1)) * alpha_sd + alpha_mu
        unc_k = alpha_smp

        penalty = torch.distributions.Normal(alpha_mu, alpha_sd).log_prob(alpha_smp)

        random_n = torch.randn(size=(self.n_covar + 1, n_sample, 1))

        sum_beta = torch.zeros((n_sample, len(y)))
        for i in range(self.n_covar + 1):
            logvar = self.logvar[i]
            mu = self.mu[i]
            rd = random_n[i]
            sd = (0.5 * logvar).exp()
            beta_smp = rd * sd + mu
            if i == self.n_covar:
                sum_beta += beta_smp
            else:
                sum_beta += beta_smp * covar[i]

            penalty += torch.distributions.Normal(mu, sd).log_prob(beta_smp)

        # unc_lbd = -sum_beta  # sigma in Stan, lambda/scale in Wikipedia, scale in Torch
        # !!! Not divided by k, contrary to the article

        k_smp = 1e-7 + torch.sigmoid(unc_k) * self.k_scale                  # alpha in Stan, k/shape in Wikipedia, k/contentration in Torch
        lbd_smp = 1e-7 + torch.sigmoid(-sum_beta / k_smp) * self.lbd_scale  # sigma in Stan, lambda/scale in Wikipdedia, scale in Torch

        lls = torch.distributions.Weibull(k_smp, lbd_smp).log_prob(y).sum(axis=-1)
        lls_mean = lls.mean()
        to_min = - lls_mean + penalty.mean()
        return to_min


def run(condition):

    n_epochs = 300
    n_sample = 40
    lbd_scale = 2
    k_scale = 2
    learning_rate = 0.05
    seed = 12345
    fig_folder = f"fig/main-all/condition{condition}"

    os.makedirs(fig_folder, exist_ok=True)

    torch.manual_seed(seed)

    data = Dataset(condition=condition)

    n_obs = len(data)
    print("n observation", n_obs)

    model = Model(data.n_covar,
                  lbd_scale=lbd_scale,
                  k_scale=k_scale)

    dataloader = torch.utils.data.DataLoader(data, batch_size=len(data))

    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())

    hist_loss = []

    for _ in tqdm(range(n_epochs)):
        for batch, (covar, y) in enumerate(dataloader):
            loss = model(covar=covar, y=y, n_sample=n_sample)
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
        smp = model.sample_concentration_parameter(n_sample=10000).detach().numpy()

        mu, sd = np.mean(smp), np.std(smp)

        lower, upper = mu - 1.96 * sd, mu + 1.96 * sd

        print(f"k={mu:.3f} [{lower:.3f}, {upper:.3f}]")

        fig, ax = plt.subplots()
        sns.histplot(smp, ax=ax, stat='density', legend=False)
        ax.set_title("k")
        plt.savefig(f"{fig_folder}/k_samples.pdf")
        plt.show()

        # ------------------------------------------

        beta_dist_prm = model.beta_dist_parameters()

        for i in range(data.n_covar+1):
            if i == data.n_covar:
                cv = "0"
            else:
                cv = data.covar[i]

            key = f"beta {cv}"
            mu, sd = beta_dist_prm[i]

            fig, ax = plt.subplots()

            lower, upper = mu-1.96*sd, mu+1.96*sd

            if (lower < 0 and upper < 0) or (upper > 0 and lower > 0):
                sign = "*"
            else:
                sign = ""
            print(f"{key}={mu:.3f} [{lower:.3f}, {upper:.3f}]{sign}")
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

        # ----------------------------------- #

        c = {(0,0): "#B1DFEA", (0,1): '#55247B', (1,0): "yellow", (1,1): "#E60082"}
        n_sample = 100

        fig, ax = plt.subplots()

        covar_set = (0, 0), (0, 1), (1, 0), (1, 1)

        for covar in covar_set:

            x = torch.linspace(0.1, 10, 200)

            alpha_mu, alpha_logvar = model.alpha_param
            alpha_sd = (0.5 * alpha_logvar).exp()
            alpha_smp = torch.randn(size=(n_sample, 1)) * alpha_sd + alpha_mu
            unc_k = alpha_smp

            random_n = torch.randn(size=(model.n_covar + 1, n_sample, 1))

            sum_beta = torch.zeros((n_sample, 1))
            for i in range(model.n_covar + 1):
                logvar = model.logvar[i]
                mu = model.mu[i]
                rd = random_n[i]
                sd = (0.5 * logvar).exp()
                beta_smp = rd * sd + mu
                if i == model.n_covar:
                    sum_beta += beta_smp
                else:
                    sum_beta += beta_smp * covar[i]

            # unc_lbd = -sum_beta  # sigma in Stan, lambda/scale in Wikipedia, scale in Torch
            # !!! Not divided by k, contrary to the article

            k_smp = 1e-7 + torch.sigmoid(unc_k) * model.k_scale  # alpha in Stan, k/shape in Wikipedia, k/contentration in Torch
            lbd_smp = 1e-7 + torch.sigmoid(
                -sum_beta / k_smp) * model.lbd_scale  # sigma in Stan, lambda/scale in Wikipdedia, scale in Torch

            print("k shape", k_smp.shape)
            print("x shape", x.shape)
            # x = x #.unsqueeze(-1)
            print("x shape", x.shape)
            print("lbd shape", lbd_smp.shape)
            y = torch.distributions.Weibull(k_smp, lbd_smp).log_prob(x).exp()

            print(y.shape)
            x = x.squeeze().numpy()
            y = y.squeeze().numpy()

            print(x.shape)
            print(y.shape)
            for y_ in y:
                sns.lineplot(x=x, y=y_, ax=ax, linestyle='-',
                             color=c[covar], zorder=10,
                             linewidth=0.3, alpha=0.1)


        plt.savefig(f"{fig_folder}/survival.pdf")
        plt.show()



def main():

    for cond in [1, 2]:  # 1: Single column / 2: double-column
        print("#"*20)
        print("Condition", cond)
        print("#"*20)
        run(condition=cond)


if __name__ == "__main__":
    main()
