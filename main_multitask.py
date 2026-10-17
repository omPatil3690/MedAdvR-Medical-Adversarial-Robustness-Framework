"""
MedAdvR Phase 3 CLI Entrypoint: Multi-Task & Semantic Disconnect Pipeline
=========================================================================
Runs the Dual-Head (Segmentation + 3-Class Pathology Classification) and
Cross-Attention Consistency Gate framework without affecting the baseline pipeline.
"""

import argparse
import torch
from framework_multitask import FrameworkMultiTaskRun


def parse_args():
    parser = argparse.ArgumentParser(
        description="MedAdvR Phase 3: Multi-Task Semantic Disconnect Training & Defense"
    )
    parser.add_argument("--dataset_path", type=str, default="./data/all_tumors", help="Path to dataset")
    parser.add_argument("--gen_type", type=str, default="edge", choices=["edge", "intensity", "texture"])
    parser.add_argument(
        "--attack_mode",
        type=str,
        default="joint_disconnect",
        choices=["joint_disconnect", "silent_tumor", "pathology_swap"],
        help="Semantic disconnect attack objective",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--lr_model", type=float, default=1e-3)
    parser.add_argument("--lr_gen", type=float, default=1e-3)
    parser.add_argument("--pretrain_epochs", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--gen_epochs", type=int, default=2)
    parser.add_argument("--model_epochs", type=int, default=2)
    parser.add_argument("--tau_audit", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_images", action="store_true", default=True)
    parser.add_argument("--save_dir", type=str, default="outputs_multitask")
    parser.add_argument("--max_buffer_size", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print("\n========== MULTI-TASK SEMANTIC DISCONNECT CONFIG ==========")
    for k, v in vars(args).items():
        print(f"  {k:<18}: {v}")
    print("===========================================================\n")

    FrameworkMultiTaskRun(
        dataset_path=args.dataset_path,
        gen_type=args.gen_type,
        attack_mode=args.attack_mode,
        device=args.device,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr_model=args.lr_model,
        lr_gen=args.lr_gen,
        pretrain_epochs=args.pretrain_epochs,
        cycles=args.cycles,
        gen_epochs=args.gen_epochs,
        model_epochs=args.model_epochs,
        save_images=args.save_images,
        save_dir=args.save_dir,
        max_buffer_size=args.max_buffer_size,
        tau_audit=args.tau_audit,
    )


if __name__ == "__main__":
    main()
