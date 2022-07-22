from multiscope import test
from multiscope.protos import tree_pb2


def main():
  test.hello()
  node_path = tree_pb2.NodePath()
  node_path.path.extend(["test", "yay"])
  print(node_path.path)


if __name__ == "__main__":
	main()
