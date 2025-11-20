#!/usr/bin/env python3
"""
Notebook Execution Script using Papermill

This script executes a Jupyter notebook using papermill and converts the result to HTML.
It automatically checks and installs required dependencies.
"""

import argparse
import sys
import subprocess
import os
import platform
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_command_exists(command):
    """Check if a command exists in the system PATH."""
    try:
        subprocess.run(
            ['which', command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def detect_package_manager():
    """Detect the package manager based on the system."""
    if check_command_exists('apt-get'):
        return 'apt'
    elif check_command_exists('yum'):
        return 'yum'
    elif check_command_exists('dnf'):
        return 'dnf'
    else:
        return None


def install_python_package(package):
    """Install a Python package using pip."""
    logger.info(f"Installing {package}...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', package],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        logger.info(f"[OK] Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[FAILED] Failed to install {package}: {e}")
        return False


def check_and_install_dependencies():
    """Check and install required dependencies."""
    logger.info("Checking dependencies...")
    
    # Check Python version
    if sys.version_info < (3, 6):
        logger.error("Python 3.6+ is required")
        sys.exit(1)
    logger.info(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check pip
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', '--version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        logger.info("[OK] pip is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("[FAILED] pip is not available. Installing pip...")
        package_manager = detect_package_manager()
        if package_manager == 'apt':
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'python3-pip'], check=True)
        elif package_manager in ['yum', 'dnf']:
            subprocess.run(['sudo', package_manager, 'install', '-y', 'python3-pip'], check=True)
        else:
            logger.error("Cannot detect package manager. Please install pip manually.")
            sys.exit(1)
        logger.info("[OK] pip installed")
    
    # Check and install papermill
    try:
        subprocess.run(
            ['papermill', '--version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        logger.info("[OK] papermill is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("papermill not found, installing...")
        if not install_python_package('papermill'):
            sys.exit(1)
    
    # Check and install jupyter nbconvert
    try:
        subprocess.run(
            ['jupyter', 'nbconvert', '--version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        logger.info("[OK] jupyter nbconvert is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("jupyter nbconvert not found, installing...")
        if not install_python_package('jupyter'):
            sys.exit(1)
    
    logger.info("All dependencies are ready!")


def parse_env_vars(env_args):
    """Parse environment variable arguments."""
    env_vars = {}
    for env_arg in env_args:
        if '=' not in env_arg:
            logger.error(f"Invalid environment variable format: {env_arg}. Expected format: KEY=VALUE")
            sys.exit(1)
        key, value = env_arg.split('=', 1)
        env_vars[key] = value
    return env_vars


def execute_notebook(notebook_path, output_notebook_path, env_vars):
    """Execute notebook using papermill."""
    logger.info(f"Executing notebook: {notebook_path}")
    logger.info(f"Output notebook will be saved to: {output_notebook_path}")
    
    if env_vars:
        logger.info("Environment variables:")
        for key, value in env_vars.items():
            # Mask sensitive values
            masked_value = value[:4] + '*' * (len(value) - 4) if len(value) > 4 else '****'
            logger.info(f"  {key}={masked_value}")
    
    # Prepare papermill command
    cmd = [
        'papermill',
        str(notebook_path),
        str(output_notebook_path),
        '--log-output',
        '--request-save-on-cell-execute'
    ]
    
    # Set environment variables
    env = os.environ.copy()
    env.update(env_vars)
    
    logger.info("Starting notebook execution...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Execute with real-time output
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Notebook execution failed with exit code {process.returncode}")
            sys.exit(1)
        
        logger.info("[OK] Notebook execution completed successfully")
        
    except FileNotFoundError:
        logger.error("papermill command not found. Please ensure papermill is installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing notebook: {e}")
        sys.exit(1)


def convert_to_html(notebook_path, html_output_path):
    """Convert executed notebook to HTML."""
    logger.info(f"Converting notebook to HTML: {html_output_path}")
    
    # Create output directory if it doesn't exist
    html_output_dir = os.path.dirname(html_output_path)
    if html_output_dir and not os.path.exists(html_output_dir):
        os.makedirs(html_output_dir, exist_ok=True)
        logger.info(f"Created output directory: {html_output_dir}")
    
    cmd = [
        'jupyter',
        'nbconvert',
        '--to', 'html',
        '--output', os.path.basename(html_output_path),
        '--output-dir', html_output_dir if html_output_dir else '.',
        str(notebook_path)
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        
        logger.info("Conversion output:")
        print(result.stdout)
        
        # Verify HTML file was created
        if not os.path.exists(html_output_path):
            logger.error(f"HTML file was not generated at {html_output_path}")
            sys.exit(1)
        
        logger.info(f"[OK] HTML file generated successfully: {html_output_path}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert notebook to HTML: {e}")
        logger.error(f"Error output: {e.stdout}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error converting notebook: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Execute a Jupyter notebook using papermill and convert to HTML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (outputs to default directory: notebook_runner_output/)
  %(prog)s -f notebook.ipynb

  # With custom output directory
  %(prog)s -f notebook.ipynb --output-dir /path/to/output

  # With environment variables
  %(prog)s -f notebook.ipynb --output-dir /path/to/output -e NGC_API_KEY=your_key -e NVIDIA_API_KEY=your_key

  # Multiple environment variables
  %(prog)s -f notebook.ipynb --output-dir /path/to/output \\
    -e NGC_API_KEY=your_key \\
    -e NVIDIA_API_KEY=your_key \\
    -e DEPLOYMENT_OPTION=1
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        required=True,
        help='Path to the input notebook file'
    )
    
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory path. All generated files will be saved here. '
             'If not provided, defaults to notebook_runner_output/ in the script directory.'
    )
    
    parser.add_argument(
        '-e', '--env',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Environment variables to pass to the notebook (can be specified multiple times)'
    )
    
    parser.add_argument(
        '--skip-deps-check',
        action='store_true',
        help='Skip dependency check and installation'
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    notebook_path = Path(args.file)
    if not notebook_path.exists():
        logger.error(f"Notebook file not found: {notebook_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to notebook_runner_output in the script's directory
        script_dir = Path(__file__).parent.absolute()
        output_dir = script_dir / 'notebook_runner_output'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output file paths (all in the output directory)
    notebook_name = notebook_path.stem
    output_notebook_path = output_dir / f"{notebook_name}.executed.ipynb"
    html_output_path = output_dir / f"{notebook_name}.html"
    
    logger.info("=" * 60)
    logger.info("Notebook Execution Script")
    logger.info("=" * 60)
    logger.info("")
    logger.info("OUTPUT PATH INFORMATION:")
    logger.info(f"   All generated files will be saved to:")
    logger.info(f"   {output_dir.absolute()}")
    logger.info("")
    logger.info(f"   Generated files:")
    logger.info(f"   - Executed notebook: {output_notebook_path.name}")
    logger.info(f"   - HTML report: {html_output_path.name}")
    logger.info("")
    logger.info("=" * 60)
    
    # Check and install dependencies
    if not args.skip_deps_check:
        check_and_install_dependencies()
    
    # Parse environment variables
    env_vars = parse_env_vars(args.env)
    
    # Execute notebook
    try:
        execute_notebook(notebook_path, output_notebook_path, env_vars)
        
        # Convert to HTML
        convert_to_html(output_notebook_path, html_output_path)
        
        logger.info("=" * 60)
        logger.info("[OK] All operations completed successfully!")
        logger.info(f"[OK] Output directory: {output_dir}")
        logger.info(f"[OK] Executed notebook: {output_notebook_path}")
        logger.info(f"[OK] HTML output: {html_output_path}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

