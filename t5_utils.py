import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

# wandb 可选，不用也不会报错
try:
    import wandb
except ImportError:
    wandb = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_wandb(args):
    """
    如果需要用 wandb 记录实验，可以在命令行加 --use_wandb。
    不想用的话，这个函数什么都不会做。
    """
    if not getattr(args, "use_wandb", False) or wandb is None:
        return
    wandb.init(
        project="t5-sql",
        name=args.experiment_name,
        config=vars(args),
    )


def initialize_model(args):
    """
    初始化 T5 模型：
    - 如果 args.finetune 为 True：从 google-t5/t5-small 加载预训练权重
    - 否则：只用同一个 config，从头初始化参数
    """
    if args.finetune:
        print("Loading pretrained model google-t5/t5-small...")
        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    else:
        print("Initializing T5-small from scratch...")
        config = T5Config.from_pretrained("google-t5/t5-small")
        model = T5ForConditionalGeneration(config)

    # 为了省显存，开启 gradient checkpointing（可选，但对你很有用）
    try:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")
    except Exception:
        pass

    model.to(DEVICE)
    return model


def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def save_model(checkpoint_dir: str, model: torch.nn.Module, best: bool):
    """
    保存模型，用于后面继续训练或做 best 模型评估。

    train_t5.py 里约定：
      - last 模型：last_model.pt
      - best 模型：best_model.pt
    存储格式：{"model_state_dict": state_dict}
    """
    mkdir(checkpoint_dir)
    fname = "best_model.pt" if best else "last_model.pt"
    ckpt_path = os.path.join(checkpoint_dir, fname)

    state = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(state, ckpt_path)
    print(f"Saved {'BEST' if best else 'last'} model checkpoint to {ckpt_path}")


def load_model_from_checkpoint(args, best: bool):
    """
    读取已经保存的模型：
      - model_type = ft / scr
      - checkpoint 目录：checkpoints/{model_type}_experiments/{experiment_name}
      - 文件名：best_model.pt 或 last_model.pt

    返回：已经 load_state_dict 并移动到 DEVICE 的 T5 模型
    """
    model_type = "ft" if args.finetune else "scr"
    checkpoint_dir = os.path.join("checkpoints", f"{model_type}_experiments", args.experiment_name)
    fname = "best_model.pt" if best else "last_model.pt"
    ckpt_path = os.path.join(checkpoint_dir, fname)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    print(f"Loading model from checkpoint: {ckpt_path}")
    model = initialize_model(args)  # 保证结构和初始化方式一致
    state = torch.load(ckpt_path, map_location=DEVICE)

    # 兼容两种存法：直接 state_dict 或 {"model_state_dict": ...}
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    return model


def initialize_optimizer_and_scheduler(args, model, epoch_length: int):
    """
    根据 args 创建 optimizer 和 scheduler。
    epoch_length = 每个 epoch 的 step 数（len(train_loader)）
    """
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler


def initialize_optimizer(args, model):
    """
    仿照 HuggingFace 的做法，把含 LayerNorm 的参数和 bias 设为 no-weight-decay。
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999),
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer_type: {args.optimizer_type}")

    return optimizer


def initialize_scheduler(args, optimizer, epoch_length: int):
    """
    根据 args 创建 scheduler：
      - none: 不用 scheduler，返回 None
      - cosine / linear: 使用 transformers 里的 scheduler
    """
    num_training_steps = max(1, epoch_length * args.max_n_epochs)
    num_warmup_steps = max(0, epoch_length * args.num_warmup_epochs)

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        raise NotImplementedError(f"Unsupported scheduler_type: {args.scheduler_type}")


def get_parameter_names(model, forbidden_layer_types):
    """
    从 HuggingFace 抄过来的辅助函数：
    返回所有不属于 forbidden_layer_types 的参数名，用来区分是否施加 weight_decay。
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # 直接定义在模型上的 nn.Parameter
    result += list(model._parameters.keys())
    return result
