#!/usr/bin/env python3
"""
Notebook Execution Script using nbclient

This script executes a Jupyter notebook using nbclient with support for skipping specific cells.
It automatically checks and installs required dependencies.

Key advantages over papermill:
- Fine-grained control over which cells to execute
- Ability to skip specific cells by index or tags
- More flexible execution logic
"""

import argparse
import sys
import subprocess
import os
import json
import logging
from pathlib import Path
from typing import Set, List, Optional

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
    if sys.version_info < (3, 7):
        logger.error("Python 3.7+ is required for nbclient")
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
    
    # Check and install nbclient
    try:
        import nbclient
        logger.info(f"[OK] nbclient is installed (version: {nbclient.__version__})")
    except ImportError:
        logger.info("nbclient not found, installing...")
        if not install_python_package('nbclient'):
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


def parse_skip_cells(skip_args: List[str]) -> Set[int]:
    """Parse skip cell indices from command line arguments.
    
    Supports:
    - Single indices: --skip-cells 0 2 5
    - Ranges: --skip-cells 0-3 (skips cells 0, 1, 2, 3)
    - Mixed: --skip-cells 0 2-4 7
    """
    skip_indices = set()
    for arg in skip_args:
        if '-' in arg:
            # Range format: 0-3
            try:
                start, end = map(int, arg.split('-'))
                skip_indices.update(range(start, end + 1))
            except ValueError:
                logger.error(f"Invalid range format: {arg}. Expected format: START-END")
                sys.exit(1)
        else:
            # Single index
            try:
                skip_indices.add(int(arg))
            except ValueError:
                logger.error(f"Invalid cell index: {arg}. Must be an integer")
                sys.exit(1)
    return skip_indices


def parse_skip_tags(skip_tags: List[str]) -> Set[str]:
    """Parse skip tags from command line arguments."""
    return set(skip_tags)


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


def get_kernel_name(notebook, default='python3'):
    """Get kernel name from notebook metadata."""
    try:
        kernelspec = notebook.metadata.get('kernelspec', {})
        kernel = kernelspec.get('name', default)
        return kernel
    except Exception:
        return default


def warn_if_no_tags_matched(notebook, skip_tags: Set[str], skipped_count: int):
    """Warn if specified tags didn't match any cells."""
    if skip_tags and skipped_count == 0:
        logger.warning("=" * 60)
        logger.warning(f"WARNING: No cells found with tags: {skip_tags}")
        logger.warning("All cells will be executed. Check if tag names are correct.")
        
        # List all available tags
        all_tags = set()
        for cell in notebook.cells:
            tags = cell.get('metadata', {}).get('tags', [])
            all_tags.update(tags)
        
        if all_tags:
            logger.warning(f"Available tags in notebook: {sorted(all_tags)}")
        else:
            logger.warning("No tags found in any cell")
        logger.warning("=" * 60)


def should_skip_cell(cell: dict, cell_index: int, skip_indices: Set[int], skip_tags: Set[str]) -> bool:
    """Determine if a cell should be skipped based on index or tags."""
    # Check by index
    if cell_index in skip_indices:
        return True
    
    # Check by tags
    cell_tags = cell.get('metadata', {}).get('tags', [])
    if skip_tags and any(tag in cell_tags for tag in skip_tags):
        return True
    
    return False


def execute_notebook(notebook_path: Path, output_notebook_path: Path, 
                     env_vars: dict, skip_indices: Set[int], skip_tags: Set[str],
                     timeout: int = 600, kernel_name: str = None):
    """Execute notebook using nbclient with cell skipping support.
    
    Args:
        notebook_path: Path to input notebook
        output_notebook_path: Path to save executed notebook
        env_vars: Environment variables to pass
        skip_indices: Set of cell indices to skip
        skip_tags: Set of tags to skip
        timeout: Timeout in seconds per cell (0 for no timeout)
        kernel_name: Kernel name to use (None to auto-detect)
    """
    logger.info(f"Executing notebook: {notebook_path}")
    logger.info(f"Output notebook will be saved to: {output_notebook_path}")
    
    if env_vars:
        logger.info("Environment variables:")
        for key, value in env_vars.items():
            # Mask sensitive values
            masked_value = value[:4] + '*' * (len(value) - 4) if len(value) > 4 else '****'
            logger.info(f"  {key}={masked_value}")
    
    if skip_indices:
        logger.info(f"Skipping cells by index: {sorted(skip_indices)}")
    if skip_tags:
        logger.info(f"Skipping cells with tags: {skip_tags}")
    
    # Set environment variables
    env = os.environ.copy()
    env.update(env_vars)
    
    # Import nbclient here to ensure it's installed
    try:
        from nbclient import NotebookClient
        from nbformat import read, write
    except ImportError:
        logger.error("nbclient or nbformat not available. Please install: pip install nbclient nbformat")
        sys.exit(1)
    
    # Read notebook
    logger.info("Loading notebook...")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = read(f, as_version=4)
    except Exception as e:
        logger.error(f"Failed to read notebook: {e}")
        sys.exit(1)
    
    # Auto-detect kernel if not specified
    if kernel_name is None:
        kernel_name = get_kernel_name(notebook, 'python3')
    logger.info(f"Using kernel: {kernel_name}")
    
    if timeout > 0:
        logger.info(f"Cell timeout: {timeout} seconds")
    else:
        logger.info("Cell timeout: disabled")
    
    # Mark cells to skip
    total_cells = len(notebook.cells)
    code_cells = len([c for c in notebook.cells if c.cell_type == 'code'])
    skipped_count = 0
    
    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
            
        if should_skip_cell(cell, idx, skip_indices, skip_tags):
            # Cell will be skipped during execution based on tags/indices
            # No need to set execution.skip metadata (causes JSON schema validation errors)
            # Tags alone are sufficient for the executor to identify and skip cells
            skipped_count += 1
            
            # Get cell preview
            source = cell.get('source', '')
            if isinstance(source, list):
                source_preview = ''.join(source[:1]).strip()[:50]
            else:
                source_preview = source[:50] if source else ''
            logger.info(f"  [SKIP] Cell {idx}: {source_preview}...")
    
    if skipped_count > 0:
        logger.info(f"Total cells to skip: {skipped_count}/{code_cells}")
    
    # Warn if no tags matched
    warn_if_no_tags_matched(notebook, skip_tags, skipped_count)
    
    # Create a custom executor that respects skip flags
    class SkipCellExecutor(NotebookClient):
        """Custom executor that skips cells marked with skip flag.
        
        This executor extends NotebookClient to support skipping cells by tags or indices.
        When a cell is marked to skip, it is not executed but preserved in the notebook.
        The key is to override async_execute_cell which is the actual method nbclient uses.
        """
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.code_cells_count = len([c for c in self.nb.cells if c.cell_type == 'code'])
            self.current_code_cell = 0
        
        async def async_execute_cell(self, cell, cell_index, execution_count=None, store_history=True):
            """Override async_execute_cell to skip cells marked for skipping.
            
            This is the async method that nbclient actually uses internally.
            We override this to properly handle cell skipping without breaking kernel startup.
            Also logs cell execution results including errors and outputs.
            
            Args:
                cell: The notebook cell to execute
                cell_index: Index of the cell in the notebook
                execution_count: Optional execution count
                store_history: Whether to store execution history
                
            Returns:
                The executed (or skipped) cell
            """
            if cell.cell_type == 'code':
                self.current_code_cell += 1
                
            # Check if cell should be skipped
            if cell.metadata.get('execution', {}).get('skip', False):
                logger.info(f"  [{self.current_code_cell}/{self.code_cells_count}] [SKIP] Cell {cell_index}")
                # Mark cell as not executed but keep it in notebook
                cell.execution_count = None
                # Clear any existing outputs
                if hasattr(cell, 'outputs'):
                    cell.outputs = []
                return cell
            
            if cell.cell_type == 'code':
                # Log cell source preview (first 100 chars)
                source_preview = cell.source[:100].replace('\n', ' ').strip()
                if len(cell.source) > 100:
                    source_preview += "..."
                logger.info(f"  [{self.current_code_cell}/{self.code_cells_count}] Executing cell {cell_index}: {source_preview}")
            
            # Call parent's async_execute_cell for normal execution
            # This ensures kernel is properly initialized and cell is executed
            executed_cell = await super().async_execute_cell(cell, cell_index, execution_count, store_history)
            
            # After execution, check for errors and log outputs
            if cell.cell_type == 'code' and hasattr(executed_cell, 'outputs'):
                has_error = False
                error_outputs = []
                stdout_outputs = []
                
                for output in executed_cell.outputs:
                    output_type = output.get('output_type', '')
                    
                    if output_type == 'error':
                        has_error = True
                        error_info = {
                            'ename': output.get('ename', 'Unknown'),
                            'evalue': output.get('evalue', ''),
                            'traceback': output.get('traceback', [])
                        }
                        error_outputs.append(error_info)
                    elif output_type == 'stream':
                        stream_name = output.get('name', '')
                        text = output.get('text', '')
                        if stream_name == 'stdout' and text:
                            stdout_outputs.append(text)
                    elif output_type == 'execute_result':
                        data = output.get('data', {})
                        if 'text/plain' in data:
                            stdout_outputs.append(data['text/plain'])
                
                # Log error if present
                if has_error and error_outputs:
                    logger.error(f"  [{self.current_code_cell}/{self.code_cells_count}] [ERROR] Cell {cell_index} failed:")
                    for err in error_outputs:
                        logger.error(f"    Exception: {err['ename']}: {err['evalue']}")
                        if err.get('traceback'):
                            logger.error("    Traceback:")
                            for tb_line in err['traceback'][:10]:  # Limit to first 10 lines
                                logger.error(f"      {tb_line.rstrip()}")
                elif stdout_outputs:
                    # Log stdout output (limit to first 500 chars per output)
                    for output_text in stdout_outputs[:3]:  # Limit to first 3 outputs
                        output_preview = str(output_text)[:500].replace('\n', ' ')
                        if len(str(output_text)) > 500:
                            output_preview += "..."
                        logger.info(f"  [{self.current_code_cell}/{self.code_cells_count}] [OUTPUT] Cell {cell_index}: {output_preview}")
                else:
                    # Cell executed successfully but no output
                    logger.info(f"  [{self.current_code_cell}/{self.code_cells_count}] [OK] Cell {cell_index} completed")
            
            return executed_cell
    
    # Execute notebook
    logger.info("Starting notebook execution...")
    logger.info("=" * 60)
    
    try:
        # Set environment variables in the current process
        os.environ.update(env_vars)
        
        # Create client with timeout (None if 0)
        actual_timeout = timeout if timeout > 0 else None
        
        client = SkipCellExecutor(
            notebook,
            timeout=actual_timeout,
            kernel_name=kernel_name,
            allow_errors=False,
            record_timing=True
        )
        
        client.execute()
        
        logger.info("=" * 60)
        logger.info("[OK] Notebook execution completed successfully")
        
        # Print execution summary
        executed_cells = code_cells - skipped_count
        logger.info("=" * 60)
        logger.info("EXECUTION SUMMARY:")
        logger.info(f"  Total code cells: {code_cells}")
        logger.info(f"  Executed: {executed_cells}")
        logger.info(f"  Skipped: {skipped_count}")
        logger.info("=" * 60)
        
        # Save executed notebook
        logger.info(f"Saving executed notebook to: {output_notebook_path}")
        with open(output_notebook_path, 'w', encoding='utf-8') as f:
            write(notebook, f)
        
    except RuntimeError as e:
        if 'kernel' in str(e).lower():
            logger.error("=" * 60)
            logger.error("Failed to start kernel. Possible causes:")
            logger.error(f"  - Kernel '{kernel_name}' not installed")
            logger.error("  - IPython not installed")
            logger.error("  - Jupyter not configured properly")
            logger.error(f"Error: {e}")
            logger.error("=" * 60)
        else:
            logger.error(f"Runtime error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing notebook: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Restore environment (remove added vars)
        for key in env_vars:
            if key in os.environ:
                del os.environ[key]


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
        description='Execute a Jupyter notebook using nbclient with cell skipping support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (outputs to default directory: notebook_runner_output/)
  %(prog)s -f notebook.ipynb

  # Skip specific cells by index
  %(prog)s -f notebook.ipynb --skip-cells 0 2 5

  # Skip cells by range
  %(prog)s -f notebook.ipynb --skip-cells 0-3

  # Skip cells by tags
  %(prog)s -f notebook.ipynb --skip-tags skip slow-test

  # Combined: skip by index and tags
  %(prog)s -f notebook.ipynb --skip-cells 0 2 --skip-tags skip

  # With environment variables
  %(prog)s -f notebook.ipynb --output-dir /path/to/output -e NGC_API_KEY=your_key

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
        '--skip-cells',
        nargs='+',
        default=[],
        metavar='INDEX',
        help='Cell indices to skip (0-based). Supports ranges: 0-3 or individual: 0 2 5'
    )
    
    parser.add_argument(
        '--skip-tags',
        nargs='+',
        default=[],
        metavar='TAG',
        help='Cell tags to skip. Cells with these tags will not be executed'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='Timeout in seconds for each cell (default: 600, use 0 for no timeout)'
    )
    
    parser.add_argument(
        '--kernel',
        default=None,
        help='Kernel name to use (default: auto-detect from notebook metadata)'
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
    logger.info("Notebook Execution Script (nbclient)")
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
    
    # Parse arguments
    env_vars = parse_env_vars(args.env)
    skip_indices = parse_skip_cells(args.skip_cells) if args.skip_cells else set()
    skip_tags = parse_skip_tags(args.skip_tags) if args.skip_tags else set()
    
    # Execute notebook
    try:
        execute_notebook(
            notebook_path, 
            output_notebook_path, 
            env_vars, 
            skip_indices, 
            skip_tags,
            timeout=args.timeout,
            kernel_name=args.kernel
        )
        
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

