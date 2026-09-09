"""npc-init — scaffold an npc_team/ in the current directory."""
import sys
import argparse
from npcpy.npc_compiler import initialize_npc_project


def main():
    parser = argparse.ArgumentParser(description="Initialize an NPC team project")
    parser.add_argument("directory", nargs="?", default=".")
    parser.add_argument("-ctx", "--context", type=str, default=None)
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("-pr", "--provider", type=str, default=None)
    args = parser.parse_args()

    initialize_npc_project(
        directory=args.directory,
        context=args.context,
        model=args.model,
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
