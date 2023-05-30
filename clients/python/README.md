# Python Client to Multiscope

This directory contains a python client to multiscope. It is in early
stage development.

The following sections describe how to set up your machine to run the
example code as well as be able to contribute to the development.

## How to Run

Before a multiscope example can be run, we need to cover:

1. Setting up the development environment,
2. Pre-conditions for each run.

We first give a quick overview of these.

Setting up the development environment:

1. Install pipenv,
2. Ensure you can compile protos into python, go,
3. Ensure you can build the go multiscope server.

For details, see below and in the main `multiscope/README.md` file.

Pre-conditions for each run:

1. The UI is of the latest version; `go generate generate.go` from the **main** multiscope folder to update it.
2. The multiscope server binary at `~/bin/multiscope_server` is of the latest version; use `go build -o ~/bin/multiscope_server multiscope.go` from the `multiscope/server` directory to rebuild it.
3. The python compiled protos are up-to-date; if not, run `multiscope/generate_python_protos.sh`.
4. You are in the correct pipenv shell; one started in the `multiscope/clients/python` directory.

If you need to do (1), you also need to do (2).

To run an example, run `multiscope/clients/python/examples/ticker.py` with `python`. This will print the http port to view multiscope under.

The `examples` directory contains more examples as well.

## Setup

The project uses Pipenv to manage the virtual environment and dependencies.
There is a troubleshooting section below that addresses common problems.

### Pipenv

To install pipenv, do the following or a variation thereof:

```sh
pip install pipenv
```

Then, navigate to the multiscope/clients/python directory to enter (activate)
the right virtual environment. If needed, pipenv will create the right
virtual environment based on the Pipfile.

```sh
# in multiscope/clients/python
pipenv shell
```

Finally, install the necessary dependencies. These will be automatically read
out from the Pipfile.

```sh
pipenv install --dev  # or, if not developing, simply: pipenv install
```

To read more about pipenv, see https://realpython.com/pipenv-guide/.

### Import System

During development, this ('multiscope-client') package is pip installed
locally so that import paths are resolved correctly.

Use absolute imports only. See the source for examples.

### Protobufs

The proto files are declared outside of the client directory, in
`multiscope/protos/`. Python code is generated from these protos into
the `multiscope/clients/python/multiscope/protos/` directory; when the protos
change this code needs to be regenerated.

The `multiscope/generate_python_protos.sh` script takes care of this.

protoc version >= 3.19.0 is required. See
https://grpc.io/docs/protoc-installation/ for installing or updating it.

### Troubleshooting

Some common sources of problems and what to do with them.

* Confusion around virtual environments:

  * The pipenv shell has to be entered in the right directory. Virtual envs
    are found based on path and name of directory. You can change directory
    afterwards.
  * `pipenv --where` will show you which directory the shell (virtual env)
    was started in.
  * You can always delete the current virtual environment if it seems to
    be corrupted someone and simply re-create it.
