import argparse
from agent import Agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the saved Flappy Bird DQN model")
    parser.add_argument(
        "--steps",
        type=int,
        default=1_000,
        help="Maximum total environment steps to run (default: 1000)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run without opening the game window",
    )
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be greater than zero")

    agent = Agent(param_set="FlappyBird-v0")
    agent.run(is_training=False, render=not args.no_render, max_steps=args.steps)
