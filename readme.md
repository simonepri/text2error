<h1 align="center">
  <b>text2error</b>
</h1>
<p align="center">
  <!-- Lint -->
  <a href="https://github.com/simonepri/text2error/actions?query=workflow:lint+branch:master">
    <img src="https://github.com/simonepri/text2error/workflows/lint/badge.svg?branch=master" alt="Lint status" />
  </a>
  <!-- Test - macOS -->
  <a href="https://github.com/simonepri/text2error/actions?query=workflow:test-macos+branch:master">
    <img src="https://github.com/simonepri/text2error/workflows/test-macos/badge.svg?branch=master" alt="Test macOS status" />
  </a>
  <!-- Test - Ubuntu -->
  <a href="https://github.com/simonepri/text2error/actions?query=workflow:test-ubuntu+branch:master">
    <img src="https://github.com/simonepri/text2error/workflows/test-ubuntu/badge.svg?branch=master" alt="Test Ubuntu status" />
  </a>
  <br />
  <!-- Code style -->
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style" />
  </a>
  <!-- Linter -->
  <a href="https://github.com/PyCQA/pylint">
    <img src="https://img.shields.io/badge/linter-pylint-ce963f.svg" alt="Linter" />
  </a>
  <!-- Types checker -->
  <a href="https://github.com/PyCQA/pylint">
    <img src="https://img.shields.io/badge/types%20checker-mypy-296db2.svg" alt="Types checker" />
  </a>
  <!-- Test runner -->
  <a href="https://github.com/pytest-dev/pytest">
    <img src="https://img.shields.io/badge/test%20runner-pytest-449bd6.svg" alt="Test runner" />
  </a>
  <!-- Task runner -->
  <a href="https://github.com/illBeRoy/taskipy">
    <img src="https://img.shields.io/badge/task%20runner-taskipy-abe63e.svg" alt="Task runner" />
  </a>
  <!-- Build tool -->
  <a href="https://github.com/python-poetry/poetry">
    <img src="https://img.shields.io/badge/build%20system-poetry-4e5dc8.svg" alt="Build tool" />
  </a>
  <br />
  <!-- License -->
  <a href="https://github.com/simonepri/text2error/tree/master/license">
    <img src="https://img.shields.io/github/license/simonepri/text2error.svg" alt="Project license" />
  </a>
</p>
<p align="center">
  〰 Introduce errors in error free text
</p>

## Synopsis

Introduce errors in error free text.


## Development

You can install this library locally for development using the commands below.
If you don't have it already, you need to install [poetry](https://python-poetry.org/docs/#installation) first.

```bash
# Clone the repo
git clone https://github.com/simonepri/text2error
# CD into the created folder
cd text2error
# Create a virtualenv and install the required dependencies using poetry
poetry install
```

You can then run commands inside the virtualenv by using `poetry run COMMAND`.  
Alternatively, you can open a shell inside the virtualenv using `poetry shell`.


If you wish to contribute to this project, run the following commands locally before opening a PR and check that no error is reported (warnings are fine).

```bash
# Run the code formatter
poetry run task format
# Run the linter
poetry run task lint
# Run the static type checker
poetry run task types
# Run the tests
poetry run task tests
```


## Authors

- **Simone Primarosa** - [simonepri][github:simonepri]

See also the list of [contributors][contributors] who participated in this project.


## License

This project is licensed under the MIT License - see the [license][license] file for details.



<!-- Links -->

[start]: https://github.com/simonepri/text2error#start-of-content
[license]: https://github.com/simonepri/text2error/tree/master/license
[contributors]: https://github.com/simonepri/text2error/contributors

[github:simonepri]: https://github.com/simonepri
