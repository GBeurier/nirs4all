"""
Main CLI entry point for nirs4all.
"""

import argparse
import sys
from .test_install import test_installation, full_test_installation, test_integration


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='nirs4all',
        description='NIRS4ALL - Near-Infrared Spectroscopy Analysis Tool'
    )

    parser.add_argument(
        '-test_install', '--test-install',
        action='store_true',
        help='Test basic installation and show dependency versions'
    )

    parser.add_argument(
        '-full_test_install', '--full-test-install',
        action='store_true',
        help='Full installation test including TensorFlow functionality'
    )

    parser.add_argument(
        '-test_integration', '--test-integration',
        action='store_true',
        help='Run integration test with sample data pipeline'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.0.3'
    )

    args = parser.parse_args()

    if args.test_install:
        result = test_installation()
        sys.exit(0 if result else 1)
    elif args.full_test_install:
        result = full_test_installation()
        sys.exit(0 if result else 1)
    elif args.test_integration:
        result = test_integration()
        sys.exit(0 if result else 1)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
