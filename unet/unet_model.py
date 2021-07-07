""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import pytorch_lightning as pl
from .loss import supported_loss, axis_std, axis_proj_loss


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class CustomUNet(pl.LightningModule):
    def __init__(
            self,
            num_channels: int = 1,
            num_classes: int = 1,
            filters: int = 16,
            num_layers: int = 4,
            loss: str = 'bce',
            bilinear: bool = False,
            learning_rate: float = 0.01,
            dropout_proba: float = 0.0,
            hidden_channels: int = 4,
            table_unet: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.num_layers = num_layers
        self.filters = filters
        self.learning_rate = learning_rate
        self.loss_name = loss
        self.axis_std_reg_coef = 1
        self.sum_reg_coef = 1
        self.axis_proj_loss_coef = 1

        self.inc = DoubleConv(self.num_channels, self.filters)
        self.output_activation = nn.Sigmoid()
        if table_unet:
            _Up = GridUp
            self.outc = OutDoubleConv(self.filters, hidden_channels, self.num_classes, dropout_proba)
        else:
            _Up = Up
            self.outc = OutConv(self.filters, hidden_channels)

        self.down_list = nn.ModuleList()
        self.up_list = nn.ModuleList()

        factor = 2
        in_channels = self.filters
        for i in range(self.num_layers):
            down = Down(in_channels, in_channels * factor)
            up = _Up(in_channels * factor, in_channels, dropout_proba=dropout_proba, bilinear=self.bilinear)

            self.down_list.append(down)
            self.up_list.append(up)
            in_channels *= factor
        self.up_list = self.up_list[::-1]

    def conv(self, x):
        emb_down_list = []
        x = self.inc(x)
        for down in self.down_list:
            emb_down_list.append(x)
            x = down(x)
        emb_down_list = emb_down_list[::-1]
        return x, emb_down_list

    def deconv(self, x, emb_down_list):
        emb_up_list = []
        for up, emb in zip(self.up_list, emb_down_list):
            emb_up_list.append(x)
            x = up(x, emb)
        return x, emb_up_list

    def forward(self, x):
        emb, emb_list = self.conv(x)
        emb, emb_list = self.deconv(emb, emb_list)
        logits = self.outc(emb)
        pred = self.output_activation(logits)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        emb, emb_down_list = self.conv(x)
        emb, _ = self.deconv(emb, emb_down_list)
        logits = self.outc(emb)
        y_hat = self.output_activation(logits)
        loss = \
            supported_loss[self.loss_name](y_hat, y)\
            - axis_std(y_hat) * self.axis_reg_coef\
            + axis_proj_loss(y_hat, y) * self.axis_proj_loss_coef\
            + y_hat.mean() * self.sum_reg_coef

        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        tensorboard_logs = {
            'train_loss': loss,
        }

        result = {
            'loss': loss,
            'log': tensorboard_logs,
        }
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        emb, emb_down_list = self.conv(x)
        emb, _ = self.deconv(emb, emb_down_list)
        logits = self.outc(emb)
        y_hat = self.output_activation(logits)
        loss = supported_loss[self.loss_name](y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tensorboard_logs = {
            'val_loss': loss,
        }

        result = {
            'val_loss': loss,
            'log': tensorboard_logs,
        }
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            verbose=True,
            factor=0.5,
            patience=2,
        ),
            'monitor': 'val_loss',
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1}

        return [optimizer], [lr_scheduler]

