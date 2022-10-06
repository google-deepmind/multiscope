# Generates all the python proto files.
BASE=$( dirname -- "$( readlink -f -- "$0"; )"; )
cd "$BASE/clients/python"
pipenv run python -m grpc_tools.protoc -I../../protos \
  --python_out=multiscope/protos --grpc_python_out=multiscope/protos \
  ../../protos/base.proto \
	../../protos/root.proto \
	../../protos/scalar.proto \
	../../protos/table.proto \
	../../protos/text.proto \
	../../protos/ticker.proto \
	../../protos/tree.proto \
	../../protos/ui.proto

echo "Finished."

# TODO(dsz): currently the set of protos need to be declared and maintained
# in both this and the generate.go files. Unify.
