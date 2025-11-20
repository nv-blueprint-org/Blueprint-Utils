# Blueprint-Utils

Utility scripts and tools for NVIDIA Blueprint projects.

## Contents

### notebook_runner/

Automated Jupyter notebook execution scripts with support for cell skipping, environment variables, and HTML generation.

Two execution backends available:
- **notebook_runner_nbclient.py** - Uses nbclient (recommended)
- **notebook_runner_papermill.py** - Uses papermill

Quick start:
```bash
cd notebook_runner
python notebook_runner_nbclient.py -f your_notebook.ipynb
```

See [notebook_runner/README.md](notebook_runner/README.md) for detailed documentation.

## Requirements

- Python 3.7+ (for nbclient) or Python 3.6+ (for papermill)
- pip

## License

See LICENSE file for details.