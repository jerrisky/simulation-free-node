import os
import torch
from data.load_data import Data
from models.base import BaseModel
from lightning.pytorch.cli import LightningCLI
from utils import *

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


class CustomLightningCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def before_fit(self):
        self.trainer.logger.log_hyperparams(self.config.fit.as_dict())

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--name", type=str, required=True)  # experiment name should be provided
        parser.link_arguments("trainer.max_steps", "model.init_args.total_steps")
        # automatic linking of arguments
        parser.link_arguments("name", "trainer.logger.init_args.name")
        # parser.link_arguments("name", "trainer.default_root_dir", compute_fn=lambda x: os.path.join("logs", x))
        parser.link_arguments("data.task_type", "model.init_args.task_criterion", compute_fn=compute_task_criterion)
        parser.link_arguments("data.task_type", "model.init_args.metric_type", compute_fn=compute_metric_type)

    def before_instantiate_classes(self):
        mode = getattr(self.config, 'subcommand', None)
        if mode is None:
            return  # not running subcommand

        # ================= 🔴 核心修改 1：强制修正日志和权重保存路径 =================
        # 获取 Runner 传入的 default_root_dir (例如 ../Logs/SFNO/dataset/...)
        root_dir = self.config[mode]['trainer'].get('default_root_dir')

        # 如果指定了 root_dir，且配置了 Logger，强制将 Logger 的路径指向 root_dir
        if root_dir and 'logger' in self.config[mode]['trainer']:
            # 确保 logger 不是 False
            if self.config[mode]['trainer']['logger']:
                # 1. 强制覆盖 save_dir，使其与 runner 指定的目录一致
                self.config[mode]['trainer']['logger']['init_args']['save_dir'] = root_dir
                
                # 2. 清空 name 和 version，防止生成 'name/version_0' 这种多余子目录
                # 这样 Checkpoints 就会直接出现在 root_dir/checkpoints/ 下
                self.config[mode]['trainer']['logger']['init_args']['name'] = ""
                self.config[mode]['trainer']['logger']['init_args']['version'] = ""
        # ================= 🔴 修改结束 =================

        self.config[mode]['trainer']['gradient_clip_val'] = 1.0 if self.config[mode]['model'][
            'init_args']['method'] == 'node' or self.config[mode]['data']['dataset'] == 'uci' else 0.

        # set hidden_dim = latent_dim for uci dataset
        if self.config[mode]['data']['task_type'] == 'regression':
            assert self.config[mode]['data']['dataset'] == 'uci'
        if self.config[mode]['data']['dataset'] == 'uci':
            self.config[mode]['model']['init_args']['hidden_dim'] = self.config[mode]['model']['init_args']['latent_dim']

    def instantiate_classes(self):
        '''
        Hacks to set the data_dim, output_dim and label scaler for UCI dataset.
        Overrides the instantiate_classes method in LightningCLI.
        '''
        mode = getattr(self.config, 'subcommand', None)
        if mode is None:  # not running subcommand
            assert self.config['data']['task_type'] != 'regression', 'debug code not implemented for uci'
            return super().instantiate_classes()

        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, "data")

        if self.config[mode]['data']['dataset'] == 'uci':
            self.config[mode]['model']['init_args']['data_dim'] = self.datamodule.train_dataset.train_dim_x
            self.config[mode]['model']['init_args']['output_dim'] = self.datamodule.train_dataset.train_dim_y
            self.config_init = self.parser.instantiate_classes(self.config)
        self.model = self._get(self.config_init, "model")
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()
        if self.config[mode]['model']['init_args']['label_scaler'] is True:
            self.datamodule.normalize_y = True
            self.model.label_scaler = self.datamodule.train_dataset.scaler_y

        # print total batch size
        per_gpu = self.config[mode]['data']['batch_size']
        num_gpus = self.trainer.world_size
        total = per_gpu * num_gpus
        print(f"Using total batch size {total} = {num_gpus} x {per_gpu}")


def main(args=None):
    cli = CustomLightningCLI(model_class=BaseModel,
                             subclass_mode_model=True,
                             datamodule_class=Data,
                             save_config_kwargs={"overwrite": True},
                             run=True,
                             args=args)
    
    # ================= 🔴 核心修改 2：加载 Best Model 并重新验证 =================
    print("-" * 50)
    print("🔄 Training Finished. Checking for Best Checkpoint...")
    
    # 检查是否配置了 ModelCheckpoint 且找到了最佳模型
    if cli.trainer.checkpoint_callback and cli.trainer.checkpoint_callback.best_model_path:
        best_path = cli.trainer.checkpoint_callback.best_model_path
        print(f"✅ Found Best Model at: {best_path}")
        print("🚀 Reloading Best Model weights to run validation...")
        
        # 这一步非常关键：
        # 1. 加载 best_model_path 的权重
        # 2. 在验证集上跑一遍
        # 3. 更新 cli.trainer.callback_metrics 为该权重的指标
        cli.trainer.validate(ckpt_path='best', dataloaders=cli.datamodule.val_dataloader())
    else:
        print("⚠️ No Best Checkpoint found! Using LAST epoch metrics.")
    print("-" * 50)
    # ================= 🔴 修改结束 =================

    # 获取 Lightning 记录的所有指标 (此时如果是 Best，这里就是 Best 的值)
    metrics = cli.trainer.callback_metrics
    
    # 如果是 LDL 任务（avg_imp 模式），构造包含所有细分指标的字典返回
    if 'val/avg_imp' in metrics:
        return {
            'avg_imp': metrics['val/avg_imp'].item(),
            'Cheby': metrics.get('val/Cheby', torch.tensor(0.0)).item(),
            'Clark': metrics.get('val/Clark', torch.tensor(0.0)).item(),
            'Canbe': metrics.get('val/Canbe', torch.tensor(0.0)).item(),
            'KL': metrics.get('val/KL', torch.tensor(0.0)).item(),
            'Cosine': metrics.get('val/Cosine', torch.tensor(0.0)).item(),
            'Inter': metrics.get('val/Inter', torch.tensor(0.0)).item()
        }
    
    # 如果没找到指标（例如异常退出），返回一个默认值字典防止 auto_run 崩溃
    return {'avg_imp': -999999.0}

if __name__ == '__main__':
    res = main()
    # 打印时也会显示完整的指标字典
    print(f"Done. Result: {res}")