
import os
import random
import pickle
from collections import OrderedDict

import torch
from einops import rearrange
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class OptimizedLabelDataset(Dataset):

    def __init__(
            self,
            quality_location: str = "",
            images_loc = "/media/sda/ziga-work/datasets/vggface2/cropped",
            validation = False,
            validation_percentage: float = .05,
            trans = None,
        ):
        super().__init__()

        self.trans = trans

        with open(quality_location, "rb") as pkl_in:
            datapoints = pickle.load(pkl_in)

        datapoints = [(f, v, os.path.join(images_loc, f)) for f, v in datapoints.items()]
        datapoints.sort(key=lambda x: x[1])
        max_, min_ = max(datapoints, key=lambda x: x[1])[1], min(datapoints, key=lambda x: x[1])[1]
        datapoints = list(map(lambda x: (x[0], (x[1] - min_)/(max_ - min_), x[2]), datapoints))
        test_samples = [datapoints[i] for i in range(0, len(datapoints), int(1. / validation_percentage))]
        train_samples = list(set(datapoints) - set(test_samples))

        if not validation:
            self.items = train_samples
        else:
            self.items = test_samples
        
        random.shuffle(self.items)

        del datapoints, test_samples, train_samples

    def __getitem__(self, index):
        _, label, i_loc = self.items[index]
        return (self.trans(Image.open(i_loc).convert("RGB")), torch.tensor(label, dtype=torch.float))

    def __len__(self):
        return len(self.items)


def train_network(
        model = None,
        trans = None,
        validation_percentage = .05,
        quality_location: str = "",
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        epochs: int = 1,
        embedding_size = 512,
        use_wandb = True,
        wandb_logger = None,
        logs_per_epoch: int = 20,
        save_location: str = "",
        verbose: bool = False,
    ):

    model = torch.nn.Sequential(
        OrderedDict([
            ('base_model', model),
            ('regressor', torch.nn.Linear(in_features=embedding_size, out_features=1))
        ])
    )
    model.cuda()

    train_dataset = OptimizedLabelDataset(trans=trans, quality_location=quality_location, validation_percentage=validation_percentage)
    val_dataset = OptimizedLabelDataset(validation=True, trans=trans, quality_location=quality_location, validation_percentage=validation_percentage)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam([
        {'params': model.regressor.parameters(), 'lr': learning_rate, 'weight_decay': 1e-4},
        {'params': list(model.base_model.parameters())[-10:], 'lr': learning_rate, 'weight_decay': 1e-5},
        {'params': list(model.base_model.parameters())[-50:-10], 'lr': learning_rate * .1, 'weight_decay': 1e-6},
        {'params': list(model.base_model.parameters())[-100:-50], 'lr': learning_rate * .01, 'weight_decay': 1e-6}
    ])

    grad_scaler = torch.cuda.amp.GradScaler()

    train_loader_size = len(train_dataloader)
    log_every = train_loader_size // logs_per_epoch

    best_validation_loss = float("inf")
    for epoch in range(epochs):
        print(f" Starting epoch {epoch}/{epochs}") if verbose else ...

        model.train()
        with tqdm(initial=0, total=len(train_dataloader), disable=not verbose) as pbar:
            for i, (input_batch, label_batch) in enumerate(train_dataloader):
                
                input_batch, label_batch = input_batch.cuda(), label_batch.cuda()

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    preds = model(input_batch)
                    loss = loss_function(preds.squeeze(), label_batch)

                grad_scaler.scale(loss).backward() 
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                grad_scaler.step(optimizer) 
                grad_scaler.update()

                if use_wandb:
                    wandb_logger.log({"train_loss": loss.item()})

                pbar.set_description(f"batch_loss: {loss.item(): .4f}")
                pbar.update(1)

                if (i+1) % log_every == 0:
                    model.eval()
                    epoch_val_loss = 0.
                    for (input_batch, label_batch) in tqdm(validation_dataloader, disable=not verbose):

                        input_batch, label_batch = input_batch.cuda(), label_batch.cuda()

                        with torch.no_grad():
                            preds = model(input_batch)

                        loss = loss_function(preds.squeeze(), label_batch)
                        epoch_val_loss += loss.item()

                    if use_wandb:
                        wandb_logger.log({"val_loss": epoch_val_loss / len(validation_dataloader)})

                    if epoch_val_loss < best_validation_loss:
                        best_validation_loss = epoch_val_loss
                        torch.save(model.state_dict(), save_location)
                    model.train()


if __name__ == "__main__":

    ...
