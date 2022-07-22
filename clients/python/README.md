# Python Client to Multiscope

This directory contains a python client to multiscope. It is in early
stage development.

The following sections describe how to set up your machine to run the
example code as well as be able to contribute to the development.

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


### Troubleshooting:

Some common sources of problems and what to do with them.

* Confusion around virtual environments:

  * The pipenv shell has to be entered in the right directory. Virtual envs
    are found based on path and name of directory. You can change directory
    afterwards.
  * `pipenv --where` will show you which directory the shell (virtual env)
    was started in.
  * You can always delete the current virtual environment if it seems to
    be corrupted someone and simply re-create it.


## Run an Example

From the pipenv shell, run `python examples/foo.py` from the
`multiscope/clients/python` directory, or an equivalent variation. At this
point this will simply print "Hello!" to the terminal.

