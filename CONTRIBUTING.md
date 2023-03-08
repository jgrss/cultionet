## Install Cultionet

### Clone the latest repo

```commandline
git clone https://github.com/jgrss/cultionet.git
```

### Create a virtual environment

```commandline
pyenv virtualenv 3.8.15 venv.cnet
```

### Activate the virtual environment

```commandline
pyenv activate venv.cnet
```

### Install

```commandline
(venv.cnet) cd cultionet/
(venv.cnet) pip install -e .[test]
```

## Create a new branch for your contribution

```commandline
(venv.cnet) git checkout -b new_branch_name
```

## After making changes, run tests

```commandline
(venv.cnet) cd tests/
(venv.cnet) pytest
```
