protoc --proto_path=protos --python_out=clients/python/multiscope/protos \
	protos/base.proto \
	protos/root.proto \
	protos/scalar.proto \
	protos/table.proto \
	protos/text.proto \
	protos/ticker.proto \
	protos/tree.proto \
	protos/ui.proto

# TODO(dsz): currently the set of protos need to be declared and maintained
# in both this and the generate.go files. Unify.
