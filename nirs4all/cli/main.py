"""
Main CLI entry point for nirs4all.
"""

import argparse
import sys


def get_version():
    """Get the current version of nirs4all."""
    try:
        # First try to get from pyproject.toml via importlib.metadata
        from importlib.metadata import version
        return version("nirs4all")
    except ImportError:
        try:
            # Fallback to package __version__
            from .. import __version__
            return __version__
        except ImportError:
            try:
                # Final fallback to pkg_resources
                import pkg_resources
                return pkg_resources.get_distribution("nirs4all").version
            except Exception:
                return "unknown"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='nirs4all',
        description='NIRS4ALL - Near-Infrared Spectroscopy Analysis Tool'
    )

    parser.add_argument(
        '-test_install', '-test-install', '--test-install', '--test_install',
        action='store_true',
        help='Test basic installation and show dependency versions'
    )

    parser.add_argument(
        '-test_integration', '-test-integration', '--test-integration', '--test_integration',
        action='store_true',
        help='Run integration test with sample data pipeline'
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {get_version()}'
    )

    args = parser.parse_args()

    if args.test_install:
        from .test_install import test_installation
        result = test_installation()
        sys.exit(0 if result else 1)
    elif args.test_integration:
        from .test_install import test_integration
        result = test_integration()
        sys.exit(0 if result else 1)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
