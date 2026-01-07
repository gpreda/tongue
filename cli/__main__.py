"""Entry point for tongue CLI client."""

import argparse
import sys

from cli.api_client import TongueAPIClient
from cli.console import ConsoleUI


def main():
    parser = argparse.ArgumentParser(description='Tongue - Spanish translation practice')
    parser.add_argument(
        '--server',
        default='http://localhost:8000',
        help='Server URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--user',
        default='default',
        help='User ID (default: default)'
    )
    args = parser.parse_args()

    client = TongueAPIClient(base_url=args.server, user_id=args.user)
    ui = ConsoleUI(client)

    try:
        ui.run()
    except KeyboardInterrupt:
        print('\nGoodbye!')
        sys.exit(0)


if __name__ == '__main__':
    main()
