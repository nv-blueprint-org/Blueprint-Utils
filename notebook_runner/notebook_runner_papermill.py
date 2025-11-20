#!/usr/bin/env python3
"""
Notebook Execution Script using Papermill

This script executes a Jupyter notebook using papermill and converts the result to HTML.
It automatically checks and installs required dependencies.

Features:
- Execute notebooks with papermill
- Skip cells by tags (e.g., --skip-tags skip)
- Pass environment variables to notebooks
- Convert executed notebooks to HTML
"""

import argparse
import sys
import subprocess
import os
import platform
import logging
import json
import shutil
from pathlib import Path
from typing import Set, List

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


def parse_skip_tags(skip_tags: List[str]) -> Set[str]:
    """Parse skip tags from command line arguments."""
    return set(skip_tags)


def validate_notebook(notebook_path: Path) -> bool:
    """Validate that notebook file is valid JSON and has required structure."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Check required fields
        if 'cells' not in nb:
            logger.error(f"Invalid notebook: missing 'cells' field")
            return False
        
        if not isinstance(nb['cells'], list):
            logger.error(f"Invalid notebook: 'cells' must be a list")
            return False
        
        if len(nb['cells']) == 0:
            logger.warning(f"Notebook is empty (no cells)")
        
        logger.info(f"[OK] Notebook is valid ({len(nb['cells'])} cells)")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid notebook JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to validate notebook: {e}")
        return False


def warn_if_no_tags_matched(notebook: dict, skip_tags: Set[str], skipped_count: int):
    """Warn if specified tags didn't match any cells."""
    if skip_tags and skipped_count == 0:
        logger.warning("=" * 60)
        logger.warning(f"WARNING: No cells found with tags: {skip_tags}")
        logger.warning("All cells will be executed. Check if tag names are correct.")
        
        # List all available tags
        all_tags = set()
        for cell in notebook.get('cells', []):
            tags = cell.get('metadata', {}).get('tags', [])
            all_tags.update(tags)
        
        if all_tags:
            logger.warning(f"Available tags in notebook: {sorted(all_tags)}")
        else:
            logger.warning("No tags found in any cell")
        logger.warning("=" * 60)


def preprocess_notebook_for_skipping(notebook_path: Path, skip_tags: Set[str], temp_notebook_path: Path) -> int:
    """Preprocess notebook to skip cells with specified tags.
    
    IMPORTANT: This is a WORKAROUND since papermill doesn't natively support skipping cells by tags.
    
    How it works:
    1. We temporarily change cells with skip tags from 'code' to 'raw' type
    2. Papermill will not execute 'raw' cells (they are treated as non-executable)
    3. After execution, we restore the original cell types
    
    Limitations:
    - This is not an official papermill feature
    - Papermill may still "process" raw cells (though it won't execute code in them)
    - For more reliable cell skipping, consider using notebook_runner_nbclient.py instead
    
    Returns:
        Number of cells that will be skipped
    """
    if not skip_tags:
        # No tags to skip, just copy the notebook
        shutil.copy2(notebook_path, temp_notebook_path)
        return 0
    
    logger.info(f"Preprocessing notebook to skip cells with tags: {skip_tags}")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read notebook: {e}")
        sys.exit(1)
    
    skipped_count = 0
    cells = notebook.get('cells', [])
    
    for idx, cell in enumerate(cells):
        cell_tags = cell.get('metadata', {}).get('tags', [])
        
        # Check if cell has any of the skip tags
        if any(tag in cell_tags for tag in skip_tags):
            # Change cell_type to 'raw' so papermill will skip it
            original_type = cell.get('cell_type', 'code')
            cell['cell_type'] = 'raw'
            cell['metadata']['original_cell_type'] = original_type
            cell['metadata']['skipped_by_tag'] = True
            skipped_count += 1
            
            # Get cell preview
            source = cell.get('source', '')
            if isinstance(source, list):
                source_preview = ''.join(source[:1]).strip()[:50]
            else:
                source_preview = source[:50] if source else ''
            
            logger.info(f"  [SKIP] Cell {idx} (tagged: {[t for t in cell_tags if t in skip_tags]}): {source_preview}...")
    
    if skipped_count > 0:
        logger.info(f"Total cells to skip: {skipped_count}/{len(cells)}")
    
    # Warn if no tags matched
    warn_if_no_tags_matched(notebook, skip_tags, skipped_count)
    
    # Save preprocessed notebook
    try:
        with open(temp_notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save preprocessed notebook: {e}")
        sys.exit(1)
    
    return skipped_count


def execute_notebook(notebook_path, output_notebook_path, env_vars, skip_tags=None):
    """Execute notebook using papermill.
    
    Args:
        notebook_path: Path to input notebook
        output_notebook_path: Path to save executed notebook
        env_vars: Environment variables to pass
        skip_tags: Set of tags to skip (cells with these tags will be skipped)
    """
    logger.info(f"Executing notebook: {notebook_path}")
    logger.info(f"Output notebook will be saved to: {output_notebook_path}")
    
    # Track original notebook path for restore
    original_notebook_path = notebook_path
    
    if env_vars:
        logger.info("Environment variables:")
        for key, value in env_vars.items():
            # Mask sensitive values
            masked_value = value[:4] + '*' * (len(value) - 4) if len(value) > 4 else '****'
            logger.info(f"  {key}={masked_value}")
    
    # Preprocess notebook if skip_tags are specified
    actual_input_notebook = notebook_path
    temp_notebook_path = None
    
    if skip_tags:
        # WORKAROUND: Papermill doesn't support --skip-tags natively
        # We preprocess the notebook by changing skip-tagged cells to 'raw' type
        # Papermill will skip executing raw cells
        logger.warning("Using workaround: Papermill doesn't natively support skipping cells by tags.")
        logger.warning("Cells with skip tags will be temporarily changed to 'raw' type.")
        logger.warning("For more reliable cell skipping, consider using notebook_runner_nbclient.py")
        
        temp_notebook_path = notebook_path.parent / f".temp_{notebook_path.name}"
        skipped_count = preprocess_notebook_for_skipping(notebook_path, skip_tags, temp_notebook_path)
        actual_input_notebook = temp_notebook_path
        
        if skipped_count > 0:
            logger.info(f"Using preprocessed notebook with {skipped_count} cells marked for skipping")
    
    # Prepare papermill command
    cmd = [
        'papermill',
        str(actual_input_notebook),
        str(output_notebook_path),
        '--log-output',
        '--request-save-on-cell-execute'
    ]
    
    # Set environment variables
    env = os.environ.copy()
    env.update(env_vars)
    
    logger.info("Starting notebook execution...")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 60)
    
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
            raise RuntimeError(f"Papermill execution failed with exit code {process.returncode}")
        
        logger.info("=" * 60)
        logger.info("[OK] Notebook execution completed successfully")
        
        # Restore original cell types in output notebook if we preprocessed
        if temp_notebook_path and temp_notebook_path.exists():
            try:
                restore_skipped_cells_in_output(output_notebook_path, original_notebook_path, skip_tags)
            except Exception as e:
                logger.warning(f"Failed to restore skipped cells in output: {e}")
        
        # Print execution summary
        try:
            with open(output_notebook_path, 'r', encoding='utf-8') as f:
                executed_nb = json.load(f)
            
            total_code_cells = len([c for c in executed_nb.get('cells', []) if c.get('cell_type') == 'code'])
            skipped = len([c for c in executed_nb.get('cells', []) 
                          if c.get('metadata', {}).get('skipped_by_tag', False)])
            executed = total_code_cells - skipped
            
            logger.info("=" * 60)
            logger.info("EXECUTION SUMMARY:")
            logger.info(f"  Total code cells: {total_code_cells}")
            logger.info(f"  Executed: {executed}")
            logger.info(f"  Skipped: {skipped}")
            logger.info("=" * 60)
        except Exception as e:
            logger.warning(f"Could not generate execution summary: {e}")
        
    except FileNotFoundError:
        logger.error("papermill command not found. Please ensure papermill is installed.")
        raise
    except Exception as e:
        logger.error(f"Error executing notebook: {e}")
        raise
    finally:
        # Always clean up temporary notebook
        if temp_notebook_path and temp_notebook_path.exists():
            try:
                temp_notebook_path.unlink()
                logger.debug(f"Cleaned up temporary notebook: {temp_notebook_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary notebook: {e}")


def restore_skipped_cells_in_output(output_notebook_path: Path, original_notebook_path: Path, skip_tags: Set[str]):
    """Restore original cell types for skipped cells in the output notebook.
    
    This ensures the output notebook maintains the original structure,
    with skipped cells marked appropriately.
    """
    if not skip_tags:
        return
    
    try:
        # Read original notebook to get cell structure
        with open(original_notebook_path, 'r', encoding='utf-8') as f:
            original_nb = json.load(f)
        
        # Read output notebook
        with open(output_notebook_path, 'r', encoding='utf-8') as f:
            output_nb = json.load(f)
        
        # Restore cell types for skipped cells
        original_cells = original_nb.get('cells', [])
        output_cells = output_nb.get('cells', [])
        
        # Safety check: ensure cell counts match
        if len(original_cells) != len(output_cells):
            logger.error(f"Cell count mismatch: original={len(original_cells)}, output={len(output_cells)}")
            logger.error("Cannot restore skipped cells safely. Output notebook may have incorrect cell types.")
            return
        
        for idx, (orig_cell, out_cell) in enumerate(zip(original_cells, output_cells)):
            cell_tags = orig_cell.get('metadata', {}).get('tags', [])
            
            if any(tag in cell_tags for tag in skip_tags):
                # Restore original cell type
                original_type = orig_cell.get('cell_type', 'code')
                out_cell['cell_type'] = original_type
                # Keep the skipped marker
                out_cell['metadata']['skipped_by_tag'] = True
                out_cell['metadata']['execution_count'] = None
                # Clear outputs if any
                if 'outputs' in out_cell:
                    out_cell['outputs'] = []
        
        # Save restored notebook
        with open(output_notebook_path, 'w', encoding='utf-8') as f:
            json.dump(output_nb, f, indent=1, ensure_ascii=False)
        
    except Exception as e:
        logger.warning(f"Failed to restore skipped cells: {e}")
        # Don't fail the whole process if restoration fails


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

  # Skip cells with specific tags
  %(prog)s -f notebook.ipynb --skip-tags skip

  # Skip cells with multiple tags
  %(prog)s -f notebook.ipynb --skip-tags skip slow-test

  # With environment variables
  %(prog)s -f notebook.ipynb --output-dir /path/to/output -e NGC_API_KEY=your_key -e NVIDIA_API_KEY=your_key

  # Combined: skip tags and environment variables
  %(prog)s -f notebook.ipynb --output-dir /path/to/output \\
    --skip-tags skip slow-test \\
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
        '--skip-tags',
        nargs='+',
        default=[],
        metavar='TAG',
        help='Cell tags to skip. Cells with these tags will not be executed (e.g., --skip-tags skip slow-test)'
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
    
    # Validate notebook
    if not validate_notebook(notebook_path):
        logger.error("Notebook validation failed")
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
    
    # Parse environment variables and skip tags
    env_vars = parse_env_vars(args.env)
    skip_tags = parse_skip_tags(args.skip_tags) if args.skip_tags else set()
    
    # Execute notebook
    try:
        execute_notebook(notebook_path, output_notebook_path, env_vars, skip_tags)
        
        # Convert to HTML
        convert_to_html(output_notebook_path, html_output_path)
        
        logger.info("=" * 60)
        logger.info("[OK] All operations completed successfully!")
        logger.info(f"[OK] Output directory: {output_dir.absolute()}")
        logger.info(f"[OK] Executed notebook: {output_notebook_path.name}")
        logger.info(f"[OK] HTML output: {html_output_path.name}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted by user")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

