# Multiscope

## Installation

Instructions to install Multiscope from git.

### Getting the code

Open a terminal on the desktop and navigate to the directory where the
multiscope directory should be created. Clone the git repository:
```
git clone -b main sso://deepmind/multiscope
```

### Getting the Tools for Building and Running

Install the protobuf compiler.

Linux:
```
sudo apt install -y protobuf-compiler
```

Mac OS:
```
brew install protobuf
```

Get the go plugins for the protobuf compiler and update the PATH so they can be found.

```
go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.28
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.2
export PATH="$PATH:$(go env GOPATH)/bin"
```

### Compile the the web assembly code

```
cd multiscope
go generate generate.go
```

### Example Run

Navigate to one of the examples and run it.

```
cd clients/go/examples/
go run double/double.go --local=false
```

Only use `--local=false` to open the HTTP port to other computers (not safe on a
public network for example).

Open the right port on the desktop in chrome to see the UI of multiscope. It
will look like this:
<img alt="double" src="doc/double.png" width="400" />

## Development

Please setup a pre-commit hook to catch errors before they are submitted:
```
cd .git/hooks
rm pre-commit.sample
chmod a+x ../../.pre-commit.git
ln -s ../../.pre-commit.git pre-commit
```

You will need to install the following executables for the pre-commit hook to
work:
```
go install golang.org/x/tools/cmd/goimports@latest
go install github.com/golangci/golangci-lint/cmd/golangci-lint@v1.46.2
```
You can test the pre-commit hook by running:
```
zsh .pre-commit.git
```
from multiscope top folder.


To push commits to the main repository:
```
git push sso://deepmind/_direct/multiscope
```
