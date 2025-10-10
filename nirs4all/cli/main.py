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

    # Minimal UI launcher options
    parser.add_argument(
        '--ui',
        action='store_true',
        help='Launch the nirs4all local web UI (development)'
    )
    parser.add_argument('--ui-host', default='127.0.0.1', help='Host for the web UI')
    parser.add_argument('--ui-port', default=8000, type=int, help='Port for the web UI')
    parser.add_argument('--ui-reload', action='store_true', help='Enable auto-reload when launching the web UI (dev only)')

    args = parser.parse_args()

    if args.test_install:
        from .test_install import test_installation
        result = test_installation()
        sys.exit(0 if result else 1)
    elif args.test_integration:
        from .test_install import test_integration
        result = test_integration()
        sys.exit(0 if result else 1)
    elif args.ui:
        # Delegate to the ui CLI module to start uvicorn
        try:
            from nirs4all.ui.cli import main as ui_main
        except Exception as e:
            print(f"❌ Cannot import nirs4all.ui.cli: {e}")
            sys.exit(2)

        # Build arg list for the ui CLI entrypoint
        ui_args = [f'--host={args.ui_host}', f'--port={args.ui_port}']
        if args.ui_reload:
            ui_args.append('--reload')

        ui_main(ui_args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()


# Backwards compatibility: older installers / console scripts imported
# `nirs4all_cli` from this module. Keep a small wrapper so those entry
# points continue to work without forcing users to reinstall.
def nirs4all_cli() -> None:
    """Legacy entrypoint wrapper that calls :func:`main`.

    Some older generated entrypoints import ``nirs4all_cli`` from
    ``nirs4all.cli.main``. Export this symbol to avoid ImportError when
    users run the console script created earlier.
    """
    main()
