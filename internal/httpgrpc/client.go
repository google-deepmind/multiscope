package httpgrpc

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"

	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

// Client is a http client sending gRPC-like request over http.
type Client struct {
	client http.Client
	addr   string
}

var _ = (grpc.ClientConnInterface)(nil)

func contentType(m proto.Message) string {
	fullName := m.ProtoReflect().Descriptor().FullName()
	return "application/protobuf;proto=" + string(fullName.Parent().Name()+"."+fullName.Name())
}

func (c *Client) post(method string, args interface{}, replyMsg proto.Message) (*http.Response, error) {
	request := args.(proto.Message)
	body, err := proto.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("cannot marshal request: %v", err)
	}
	req, err := http.NewRequest("POST", c.addr, bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("cannot issue a new request: %v", err)
	}
	req.Header.Set(_FunctionCall, method)
	req.Header.Set(_Accept, contentType(replyMsg))
	req.Header.Set(_ContentType, contentType(request))

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http request error: %v", err)
	}
	return resp, nil
}

func (c *Client) assign(method string, resp *http.Response, replyMsg proto.Message) error {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("cannot read request body: %v", err)
	}
	defer resp.Body.Close()
	contentType := resp.Header.Get(_ContentType)
	if contentType == contentError {
		return fmt.Errorf("server issued an error: %v", string(body))
	}
	if err := proto.Unmarshal(body, replyMsg); err != nil {
		return fmt.Errorf("cannot unmarshal body for type %T: %v", replyMsg, err)
	}
	return nil
}

// Invoke performs a unary RPC and returns after the response is received into reply.
func (c *Client) Invoke(ctx context.Context, method string, args, reply any, opts ...grpc.CallOption) error {
	if reply == nil {
		return fmt.Errorf("cannot invoke method %s with a nil reply", method)
	}
	replyMsg, ok := reply.(proto.Message)
	if !ok {
		return fmt.Errorf("cannot type %T to proto.Message for method %s", reply, method)
	}
	resp, err := c.post(method, args, replyMsg)
	if err != nil {
		return err
	}
	if err := c.assign(method, resp, replyMsg); err != nil {
		return err
	}
	return nil
}

// NewStream begins a streaming RPC.
func (*Client) NewStream(ctx context.Context, desc *grpc.StreamDesc, method string, opts ...grpc.CallOption) (grpc.ClientStream, error) {
	return nil, fmt.Errorf("stream not implemented over http")
}

// Connect to a given address.
func Connect(scheme, host string) *Client {
	return &Client{addr: scheme + "://" + host + "/httpgrpc"}
}
