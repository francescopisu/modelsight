# modelsight

Better insights into Machine Learning models performance

## Installation

### Create a conda environment
```bash
# Create a bootstrap env
conda create -p /tmp/bootstrap -c conda-forge mamba conda-lock poetry='1.*' python=3.10
conda activate /tmp/bootstrap

# Create Conda lock file(s) from environment.yml
conda-lock -k explicit --conda mamba
# Set up Poetry
poetry init --python=~3.10  # version spec should match the one from environment.yml

# Add conda-lock (and other packages, as needed) to pyproject.toml and poetry.lock
poetry add --lock conda-lock

# Remove the bootstrap env
conda deactivate
rm -rf /tmp/bootstrap

# Add Conda spec and lock files
git add environment.yml virtual-packages.yml conda-osx-arm64.lock
# Add Poetry spec and lock files
git add pyproject.toml poetry.lock
git commit
```

```bash
conda create --prefix ./envs/modelsight_env --file conda-osx-arm64.lock
conda activate ./envs/modelsight_env
poetry install
```

```bash
# Re-generate Conda lock file(s) based on environment.yml
conda-lock -k explicit --conda mamba
# Update Conda packages based on re-generated lock file
mamba update --file conda-osx-arm64.lock
# Update Poetry packages and re-generate poetry.lock
poetry update
```

```bash
$ pip install modelsight
```

## Usage

- TODO

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`modelsight` was created by Francesco Pisu. It is licensed under the terms of the GNU General Public License v3.0 license.

## Credits

`modelsight` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
