import wandb

def main():
    wandb.init(
        project="multimodal-nav-rl",
        name="sanity-test-run",
        config={
            "phase": "reward_conditioning",
        }
    )

    wandb.log({"test_metric": 1})

    wandb.finish()

if __name__ == "__main__":
    main()
