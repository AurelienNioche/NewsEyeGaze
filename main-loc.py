import os

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Dataset(torch.utils.data.Dataset):

    def __init__(self, condition):
        self.x_loc, self.y_loc, self.y = self.load_data(condition)

    def load_data(self, condition):
        df = pd.read_csv("data/stan_data_locmodel-exact.csv", index_col=0)
        df = df[df.x == condition]
        x_loc = torch.tensor(df.xloc.values, dtype=torch.float)
        y_loc = torch.tensor(df.yloc.values, dtype=torch.float)
        # x_loc *= 1920
        # y_loc *= 1800
        x_loc -= 0.5
        y_loc -= 0.5

        y = torch.tensor(df.y.values, dtype=torch.float)
        return x_loc, y_loc, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_loc[idx], self.y_loc[idx], self.y[idx]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(8))

    def draw_samples(self, x_loc, y_loc, n_sample=100, compute_penalty=False, lbd_scale=2, k_scale=2):

        alpha_mu, alpha_logvar, \
            beta0_mu, beta0_logvar, \
            beta_xloc_mu, beta_xloc_logvar, \
            beta_yloc_mu, beta_yloc_logvar \
            = self.param

        alpha_sd = (0.5 * alpha_logvar).exp()
        beta0_sd = (0.5 * beta0_logvar).exp()
        beta_xloc_sd = (0.5 * beta_xloc_logvar).exp()
        beta_yloc_sd = (0.5 * beta_yloc_logvar).exp()

        alpha = torch.randn(size=(n_sample, 1)) * alpha_sd + alpha_mu

        beta_0 = torch.randn(size=(n_sample, 1)) * beta0_sd + beta0_mu
        beta_xloc = torch.randn(size=(n_sample, 1)) * beta_xloc_sd + beta_xloc_mu
        beta_yloc = torch.randn(size=(n_sample, 1)) * beta_yloc_sd + beta_yloc_mu

        unc_k = alpha
        unc_lbd = -(beta_0 + beta_xloc * x_loc + beta_yloc * y_loc)  # sigma in Stan, lambda/scale in Wikipdedia, scale in Torch

        k = torch.sigmoid(unc_k) * k_scale        # alpha in Stan, k/shape in Wikipedia, k/contentration in Torch
        lbd = torch.sigmoid(unc_lbd) * lbd_scale  # sigma in Stan, lambda/scale in Wikipdedia, scale in Torch

        smp = {"k": k, "lbd": lbd,
               "beta_0": beta_0, "beta_xloc": beta_xloc, "beta_yloc": beta_yloc}
        if compute_penalty:

            penalty = 0
            for (mu, sd, samples) in ((alpha_mu, alpha_sd, alpha),
                                      (beta0_mu, beta0_sd, beta_0),
                                      (beta_xloc_mu, beta_xloc_sd, beta_xloc),
                                      (beta_yloc_mu, beta_yloc_sd, beta_yloc)):

                penalty += torch.distributions.Normal(mu, sd).log_prob(samples)

            return smp, penalty

        return smp

    def beta_dist_parameters(self):

        alpha_mu, alpha_logvar, \
            beta0_mu, beta0_logvar, \
            beta_xloc_mu, beta_xloc_logvar, \
            beta_yloc_mu, beta_yloc_logvar \
            = self.param

        # alpha_sd = (0.5 * alpha_logvar).exp()
        beta0_sd = (0.5 * beta0_logvar).exp()
        beta_xloc_sd = (0.5 * beta_xloc_logvar).exp()
        beta_yloc_sd = (0.5 * beta_yloc_logvar).exp()
        return {"beta0": (beta0_mu, beta0_sd),
                "beta_xloc": (beta_xloc_mu, beta_xloc_sd),
                "beta_yloc": (beta_yloc_mu, beta_yloc_sd)}

    def forward(self, x_loc, y_loc, y, n_sample=100):

        smp, penalty = \
            self.draw_samples(compute_penalty=True,
                              n_sample=n_sample,
                              x_loc=x_loc, y_loc=y_loc)

        lls = torch.distributions.Weibull(smp['k'], smp['lbd']).log_prob(y).sum(axis=-1)
        lls_mean = lls.mean()
        to_min = - lls_mean + penalty.mean()
        return to_min


def main():

    fig_folder = "fig/main-loc"
    os.makedirs(fig_folder, exist_ok=True)

    condition = 2

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
        for batch, (x_loc, y_loc, y) in enumerate(dataloader):
            loss = model(x_loc=x_loc, y_loc=y_loc, y=y, n_sample=n_sample)
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
        smp = model.draw_samples(n_sample=10000, x_loc=data.x_loc, y_loc=data.y_loc)

    for key, v in smp.items():
        if key == "lbd":
            v = v.mean(axis=-1)
        elif key.startswith("beta"):
            continue

        m = v.mean()
        print(f"{key}={m:.3f}")

        fig, ax = plt.subplots()
        sns.histplot(v.detach().numpy(), ax=ax, stat='density', legend=False)
        ax.set_title(key)
        plt.savefig(f"{fig_folder}/{key}_samples.pdf")
        plt.show()

    with torch.no_grad():
        beta_dist_prm = model.beta_dist_parameters()

        for key, v in beta_dist_prm.items():
            fig, ax = plt.subplots()

            mu, sd = v
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
