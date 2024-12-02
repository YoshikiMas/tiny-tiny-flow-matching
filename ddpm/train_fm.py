from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import call
from tqdm import tqdm


@hydra.main(
    version_base=None,
    config_path="train_conf",
    config_name="example_fm",
)
def main(cfg) -> None:
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])

    # setup
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dataset = call(cfg.dataset)
    dataloader = call(cfg.dataloader, dataset)
    ns = call(cfg.noise_scheduler)
    model = call(cfg.model)
    optimizer = call(cfg.optimizer, model.parameters())
    criterion = torch.nn.MSELoss(reduction="sum")

    # training
    with tqdm(total=cfg.num_epochs) as pbar:
        for _ in range(cfg.num_epochs):
            model.train()
            train_loss = 0
            for batch in dataloader:
                batch = batch[0]
                x0 = torch.randn(batch.shape)

                # NOTE: modified for conditional flow-matching
                sigma_min = 1.0e-3
                t = np.random.uniform(low=0.0, high=1.0)
                xt = t * batch + (1.0 - (1.0 - sigma_min) * t) * x0
                u_target = batch - (1.0 - sigma_min) * x0
                u_pred = model(xt, t)
                loss = criterion(u_pred, u_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(dataset)
            pbar.set_postfix({"loss": f"{train_loss:.4f}"})
            pbar.update(1)

    torch.save(model.state_dict(), log_dir / "params.pt")
    print(f"saving model to {log_dir}")


if __name__ == "__main__":
    main()
