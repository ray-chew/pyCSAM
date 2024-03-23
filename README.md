<p align="center">
  <a href="https://ray-chew.github.io/spec_appx/index.html">
  <img alt="CSAM Logo" src="https://ray-chew.github.io/spec_appx/_static/logo.png">
  </a>
</p>

<h2 align="center">Constrained Spectral Approximation Method</h2>


<p align="center">
<a href="https://github.com/ray-chew/spec_appx/actions/workflows/documentation.yml">
<img alt="GitHub Actions: docs" src=https://img.shields.io/github/actions/workflow/status/ray-chew/spec_appx/documentation.yml?logo=github&label=docs>
</a>
<a href="https://www.gnu.org/licenses/gpl-3.0">
<img alt="License: GPL v3" src=https://img.shields.io/badge/License-GPLv3-blue.svg>
</a>
<a href="https://github.com/psf/black">
<img alt="Code style: black" src=https://img.shields.io/badge/code%20style-black-000000.svg>
</a>
</p>


The Constrained Spectral Approximation Method (CSAM) is a physically sound and robust method for approximating the spectrum of subgrid-scale orography. It operates under the following constraints:

* Utilises a limited number of spectral modes (no more than 100)
* Significantly reduces the complexity of physical terrain by over 500 times
* Maintains the integrity of physical information to a large extent
* Compatible with unstructured geodesic grids
* Inherently scale-aware

This method is primarily used to represent terrain for weather forecasting purposes, but it also shows promise for broader data analysis applications.

---

**[Read the documentation here](https://ray-chew.github.io/spec_appx/index.html)**

---

## Requirements

See [`requirements.txt`](https://github.com/ray-chew/spec_appx/blob/main/requirements.txt)

> **NOTE:**  The Sphinx dependencies can be found in [`docs/conf.py`](https://github.com/ray-chew/spec_appx/blob/main/docs/source/conf.py).


## Usage

### Installation

Fork this repository and clone your remote fork.

### Configuration

The user-defined input parameters are in the [`inputs`](https://github.com/ray-chew/spec_appx/tree/main/inputs) subpackage. These parameters are imported into the run scripts in [`runs`](https://github.com/ray-chew/spec_appx/tree/main/runs). 

### Execution

A simple setup can be found in [`runs.idealised_isosceles`](https://github.com/ray-chew/spec_appx/blob/main/runs/idealised_isosceles.py). To execute this run script:

```console
python3 ./runs/idealised_isosceles.py
```

However, the codebase is structured such that the user can easily assemble a run script to define their own experiments. Refer to the documentation for the [available APIs](https://ray-chew.github.io/spec_appx/api.html).

## License

GNU GPL v3 (tentative)

## Contributions

Refer to the open issues that require attention.

Any changes, improvements, or bug fixes can be submitted to upstream via a pull request.

