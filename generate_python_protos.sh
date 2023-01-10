# Generates all the python proto files.

PIPENV=`which pipenv`

if [[ ! -x "$PIPENV" ]]; then
  printf "\t\033[41mPlease install pipenv (https://pipenv.pypa.io/en/latest/install/).\033[0m"
  exit 1
fi

# Step 1: Find the base directory (where this script is).
BASE=$( dirname -- "$( readlink -f -- "$0"; )"; )
cd "$BASE"

# Step 1.5: Ensure the temporary directory we will create and delete does not
#           already exist
if [ -d "protos/multiscope" ]
then
    echo "Directory 'protos/multiscope' already exists!"
    echo "    -- Please remedy and try again."
    exit 1
fi

# Step 2: Fetch the python virtual environment where we have the python proto
#         generation plugins installed.
cd clients/python
source $(pipenv --venv)/bin/activate
cd ../..

# Step 3: Restructure the protos slightly so that their compiled version will
#         work as we'd like them.
mkdir -p protos/multiscope/protos
cp protos/*.proto protos/multiscope/protos/
# Wow, on MacOS the equivalent sed command does not do anything (even with the)
# correct -i.bak or similar variant. Do `brew install gnu-sed` to install
# and use gnu-sed (gsed).
sed -i -E 's/import "(\w+).proto";/import "multiscope\/protos\/\1.proto";/' \
    protos/multiscope/protos/*.proto

# Step 4: Generate the protos.
python -m grpc_tools.protoc --proto_path=protos \
  --python_out=clients/python/ \
  --grpc_python_out=clients/python/ \
    protos/multiscope/protos/*.proto

# Step 5: Cleanup.
rm -rf protos/multiscope
deactivate

echo "Finished."


# pipenv run python -m grpc_tools.protoc -I../../protos \
#   --python_out=multiscope/protos --grpc_python_out=multiscope/protos \
#   ../../protos/base.proto \
# 	../../protos/root.proto \
# 	../../protos/scalar.proto \
# 	../../protos/table.proto \
# 	../../protos/text.proto \
# 	../../protos/ticker.proto \
# 	../../protos/tree.proto \
# 	../../protos/ui.proto


# Original was:
# protoc --proto_path=protos --python_out=clients/python/multiscope/protos \
#        protos/base.proto \
#        protos/root.proto \
#        protos/scalar.proto \
#        protos/table.proto \
#        protos/text.proto \
#        protos/ticker.proto \
#        protos/tree.proto \
#        protos/ui.proto
