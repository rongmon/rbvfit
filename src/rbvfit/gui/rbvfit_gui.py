#!/usr/bin/env python
"""
Command-line wrapper for rbvfit GUI
Usage: 
    rbvfit_gui                    # Launch GUI
    rbvfit_gui config.rbv         # Load configuration file
"""

import sys
import argparse
from pathlib import Path


def get_rbvfit_version():
    """Get rbvfit version, fallback to 2.0 if not available"""
    try:
        import rbvfit
        return getattr(rbvfit, '__version__', '2.0.1')
    except ImportError:
        return '2.0.1'


def main():
    """Main entry point for rbvfit GUI command-line interface"""
    parser = argparse.ArgumentParser(
        description='Launch rbvfit GUI for Voigt profile fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rbvfit_gui                    Launch GUI
  rbvfit_gui project.rbv        Load project file
  rbvfit_gui --version          Show version info
        """
    )
    
    parser.add_argument(
        'config_file', 
        nargs='?', 
        help='Optional project file (.rbv) to load on startup'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'rbvfit {get_rbvfit_version()}'
    )
    
    args = parser.parse_args()
    
    # Validate config file if provided
    if args.config_file:
        config_path = Path(args.config_file)
        
        if not config_path.exists():
            print(f"Error: Project file '{args.config_file}' not found")
            sys.exit(1)
            
        if not config_path.suffix == '.rbv':
            print(f"Warning: Expected .rbv file, got {config_path.suffix}")
        
        # Store the config file path for the GUI to pick up
        # We'll modify sys.argv so the GUI main() function can access it
        sys.argv = ['rbvfit_gui', str(config_path.resolve())]
    else:
        # Clean argv for GUI
        sys.argv = ['rbvfit_gui']
    
    try:
        # Import and launch the GUI
        from rbvfit.gui.main_window import main as gui_main
        gui_main()
        
    except ImportError as e:
        print(f"Error importing rbvfit GUI: {e}")
        print("Make sure rbvfit is properly installed with GUI dependencies.")
        print("Try: pip install rbvfit[gui]")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error launching rbvfit GUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()