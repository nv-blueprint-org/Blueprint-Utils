# Notebook Runner

Automated Jupyter notebook execution scripts with support for cell skipping, environment variables, and HTML generation.

## Overview

This directory contains two notebook execution scripts:

- **`notebook_runner_nbclient.py`** - Uses nbclient for reliable cell skipping (Recommended)
- **`notebook_runner_papermill.py`** - Uses papermill with workaround for cell skipping

Both scripts execute Jupyter notebooks and generate HTML reports automatically.

## Quick Start

### Basic Usage

Execute a notebook with default settings:

```bash
python notebook_runner_nbclient.py -f notebook.ipynb
```

Output files will be saved to `notebook_runner_output/` directory:
- `notebook.executed.ipynb` - Executed notebook
- `notebook.html` - HTML report

### Skip Cells by Tags

Skip cells that have specific tags:

```bash
python notebook_runner_nbclient.py -f notebook.ipynb --skip-tags skip slow-test
```

### Pass Environment Variables

Pass environment variables to the notebook:

```bash
python notebook_runner_nbclient.py -f notebook.ipynb \
  -e API_KEY=your_key \
  -e ENV=production
```

### Custom Output Directory

Specify custom output directory:

```bash
python notebook_runner_nbclient.py -f notebook.ipynb \
  --output-dir /path/to/output
```

### Complete Example

```bash
python notebook_runner_nbclient.py -f notebook.ipynb \
  --skip-tags skip slow-test \
  --timeout 1200 \
  --kernel python3 \
  -e NGC_API_KEY=xxx \
  -e NVIDIA_API_KEY=yyy \
  --output-dir ./results
```

## Installation

### Prerequisites

- Python 3.7+ (for nbclient) or Python 3.6+ (for papermill)
- pip

### Automatic Installation

The scripts automatically check and install required dependencies on first run:
- nbclient / papermill
- jupyter nbconvert

To skip dependency check:
```bash
python notebook_runner_nbclient.py -f notebook.ipynb --skip-deps-check
```

### Manual Installation

```bash
# For nbclient version
pip install nbclient nbformat jupyter

# For papermill version
pip install papermill jupyter
```

## Scripts Comparison

### notebook_runner_nbclient.py (Recommended)

**Advantages:**
- 100% reliable cell skipping
- Skip cells by index or tags
- Configurable timeout per cell
- Configurable kernel
- Better error messages
- Progress indicator

**Usage:**
```bash
python notebook_runner_nbclient.py -f notebook.ipynb [OPTIONS]
```

### notebook_runner_papermill.py

**Advantages:**
- Uses familiar papermill workflow
- Good for papermill users

**Limitations:**
- Cell skipping is a workaround (less reliable)
- No index-based skipping
- Fixed timeout and kernel settings

**Usage:**
```bash
python notebook_runner_papermill.py -f notebook.ipynb [OPTIONS]
```

## Command-Line Options

### Common Options (Both Scripts)

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --file` | Input notebook file (required) | - |
| `--output-dir` | Output directory | `notebook_runner_output/` |
| `-e, --env` | Environment variables (KEY=VALUE) | - |
| `--skip-tags` | Tags to skip | - |
| `--skip-deps-check` | Skip dependency check | False |

### nbclient-Only Options

| Option | Description | Default |
|--------|-------------|---------|
| `--skip-cells` | Cell indices to skip (0-based) | - |
| `--timeout` | Timeout per cell in seconds (0=no timeout) | 600 |
| `--kernel` | Kernel name to use | Auto-detect |

## Cell Tagging

### How to Add Tags to Cells

Tags are metadata stored in notebook cells, not visible by default.

#### In Jupyter Notebook
1. Go to `View` → `Cell Toolbar` → `Tags`
2. Tag input boxes will appear above each cell
3. Enter tag names (space-separated)

#### In JupyterLab
1. Right-click cell → `Add Cell Tag`
2. Or use the Property Inspector panel

#### Programmatically
Tags are stored in cell metadata:
```json
{
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "tags": ["skip", "slow-test"]
      },
      "source": ["print('Hello')"]
    }
  ]
}
```

### Tag Examples

Common tag patterns:
- `skip` - Skip execution
- `slow-test` - Long-running cells
- `debug` - Debug-only cells
- `production-only` - Production environment only
- `dev-only` - Development environment only

### Skip Cells by Index (nbclient only)

Skip specific cells by their position (0-based):

```bash
# Skip cells 0, 2, and 5
python notebook_runner_nbclient.py -f notebook.ipynb --skip-cells 0 2 5

# Skip cell range 0-3 (cells 0, 1, 2, 3)
python notebook_runner_nbclient.py -f notebook.ipynb --skip-cells 0-3

# Combined: skip by index and tags
python notebook_runner_nbclient.py -f notebook.ipynb \
  --skip-cells 0 2 \
  --skip-tags skip
```

## Examples

### Example 1: Basic Execution

```bash
python notebook_runner_nbclient.py -f analysis.ipynb
```

Output:
```
============================================================
Notebook Execution Script (nbclient)
============================================================

OUTPUT PATH INFORMATION:
   All generated files will be saved to:
   /path/to/notebook_runner_output

   Generated files:
   - Executed notebook: analysis.executed.ipynb
   - HTML report: analysis.html

============================================================
[OK] Python 3.9.0
[OK] pip is available
[OK] nbclient is installed (version: 0.7.0)
[OK] jupyter nbconvert is installed
All dependencies are ready!
[OK] Notebook is valid (10 cells)
Executing notebook: analysis.ipynb
Using kernel: python3
Cell timeout: 600 seconds
Starting notebook execution...
============================================================
  [1/8] Executing cell 0...
  [2/8] Executing cell 1...
  ...
============================================================
[OK] Notebook execution completed successfully
============================================================
EXECUTION SUMMARY:
  Total code cells: 8
  Executed: 8
  Skipped: 0
============================================================
[OK] HTML file generated successfully: analysis.html
============================================================
[OK] All operations completed successfully!
============================================================
```

### Example 2: Skip Cells with Tags

```bash
python notebook_runner_nbclient.py -f notebook.ipynb --skip-tags skip slow-test
```

Output includes:
```
Skipping cells with tags: {'skip', 'slow-test'}
  [SKIP] Cell 2: import time...
  [SKIP] Cell 5: # This is slow...
Total cells to skip: 2/10

...

EXECUTION SUMMARY:
  Total code cells: 10
  Executed: 8
  Skipped: 2
```

### Example 3: Custom Timeout and Kernel

```bash
python notebook_runner_nbclient.py -f notebook.ipynb \
  --timeout 1800 \
  --kernel ir
```

For R notebooks or long-running cells.

### Example 4: Environment Variables

```bash
python notebook_runner_nbclient.py -f notebook.ipynb \
  -e NGC_API_KEY=nvapi-xxx \
  -e NVIDIA_API_KEY=nvapi-yyy \
  -e DEPLOYMENT_OPTION=1
```

Access in notebook:
```python
import os
api_key = os.environ.get('NGC_API_KEY')
```

### Example 5: Production Deployment

```bash
python notebook_runner_nbclient.py -f workflow.ipynb \
  --skip-tags debug dev-only \
  --timeout 3600 \
  -e ENV=production \
  -e LOG_LEVEL=INFO \
  --output-dir /var/reports/$(date +%Y%m%d)
```

## Troubleshooting

### No Cells Found with Tags

If you see:
```
WARNING: No cells found with tags: {'skip'}
Available tags in notebook: ['debug', 'slow-test']
```

**Solution:** Check tag names match exactly (case-sensitive).

### Kernel Not Found

If you see:
```
Failed to start kernel. Possible causes:
  - Kernel 'python3' not installed
```

**Solution:**
```bash
# List available kernels
jupyter kernelspec list

# Install kernel
python -m ipykernel install --user --name python3

# Or specify existing kernel
python notebook_runner_nbclient.py -f notebook.ipynb --kernel <name>
```

### Cell Timeout

If cells timeout:
```bash
# Increase timeout to 30 minutes
python notebook_runner_nbclient.py -f notebook.ipynb --timeout 1800

# Disable timeout
python notebook_runner_nbclient.py -f notebook.ipynb --timeout 0
```

### Papermill Workaround Warning

When using papermill script, you'll see:
```
WARNING: Using workaround: Papermill doesn't natively support skipping cells by tags.
WARNING: For more reliable cell skipping, consider using notebook_runner_nbclient.py
```

**Solution:** Switch to nbclient version for 100% reliable cell skipping.

## Advanced Usage

### Programmatic Usage

Both scripts can be imported and used programmatically:

```python
from pathlib import Path
from notebook_runner_nbclient import execute_notebook

notebook_path = Path('notebook.ipynb')
output_path = Path('output.ipynb')
env_vars = {'API_KEY': 'xxx'}
skip_tags = {'skip', 'slow-test'}

execute_notebook(
    notebook_path,
    output_path,
    env_vars,
    skip_indices=set(),
    skip_tags=skip_tags,
    timeout=600,
    kernel_name='python3'
)
```

### CI/CD Integration

```yaml
# GitHub Actions example
name: Execute Notebooks
on: [push]

jobs:
  execute:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Execute notebook
        run: |
          python notebook_runner/notebook_runner_nbclient.py \
            -f notebooks/analysis.ipynb \
            --skip-tags slow-test \
            -e API_KEY=${{ secrets.API_KEY }} \
            --output-dir ./reports
      
      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: reports
          path: reports/
```

### Docker Usage

```dockerfile
FROM python:3.9

WORKDIR /app
COPY notebook_runner/ /app/notebook_runner/
COPY notebooks/ /app/notebooks/

RUN pip install nbclient nbformat jupyter

CMD ["python", "notebook_runner/notebook_runner_nbclient.py", \
     "-f", "notebooks/workflow.ipynb", \
     "--output-dir", "/app/output"]
```

## Output Files

### Executed Notebook (.ipynb)

The executed notebook contains:
- All cell outputs
- Execution counts
- Execution timing (if recorded)
- Metadata about skipped cells

### HTML Report (.html)

The HTML file is a standalone report containing:
- Formatted notebook content
- Code cells and outputs
- Images and plots
- Can be opened in any web browser

## Best Practices

1. **Use Tags for Conditional Execution**
   - Tag cells that should be skipped in certain environments
   - Use descriptive tag names: `skip`, `slow-test`, `debug`, etc.

2. **Set Appropriate Timeouts**
   - Default 600s is suitable for most cells
   - Increase for long-running operations
   - Use 0 to disable for very long jobs

3. **Manage Secrets Safely**
   - Pass sensitive data via environment variables
   - Never hardcode secrets in notebooks
   - Use `-e` flag to inject secrets at runtime

4. **Validate Before Production**
   - Test with `--skip-tags dev-only debug`
   - Use tags to separate dev and prod code paths

5. **Use nbclient for Reliability**
   - Prefer `notebook_runner_nbclient.py` for production
   - Papermill version uses workarounds for cell skipping

## Performance Tips

- Skip unnecessary cells using tags
- Use `--timeout 0` for long-running notebooks
- Skip dependency check with `--skip-deps-check` after first run
- Use `--skip-cells` to skip by index for faster execution planning

## Migration from Papermill

If you're using the papermill script, migration is seamless:

```bash
# Old command
python notebook_runner_papermill.py -f notebook.ipynb --skip-tags skip

# New command (drop-in replacement)
python notebook_runner_nbclient.py -f notebook.ipynb --skip-tags skip
```

Additional benefits:
- More reliable cell skipping
- Configurable timeout and kernel
- Skip by cell index
- Better progress indication

## Support and Documentation

- For detailed script comparison, see `SCRIPTS_COMPARISON.md`
- For issues, check the Troubleshooting section
- Both scripts have `--help` for quick reference

## License

See project root for license information.

