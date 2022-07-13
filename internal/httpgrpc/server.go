package httpgrpc

import (
	"context"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"reflect"
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

type (
	// Registerer registers a gRPC server.
	Registerer func(*grpc.Server)

	// Service provides a gRPC service.
	Service interface {
		Desc() Registerer
	}

	// Server serves gRPC-like requests over http.
	Server struct {
		registry map[string]Service
	}
)

// NewServer creates a server serving gRPC-like request on top of http.
func NewServer() *Server {
	return &Server{
		registry: make(map[string]Service),
	}
}

func toDesc(ep Service) map[string]grpc.ServiceInfo {
	s := grpc.NewServer()
	ep.Desc()(s)
	return s.GetServiceInfo()
}

// Register a gRPC service to the server.
func (s *Server) Register(service Service) error {
	for name := range toDesc(service) {
		s.registry[name] = service
	}
	return nil
}

const contentError = "text/plain; charset=utf-8; error"

func wErr(w http.ResponseWriter, format string, a ...interface{}) {
	fmt.Println("error:", fmt.Sprintf(format, a...))
	w.Header().Set(_ContentType, contentError)
	_, err := w.Write([]byte(fmt.Sprintf(format, a...)))
	if err != nil {
		log.Print(err)
	}
}

func protoFromMethod(service Service, met reflect.Value) (proto.Message, error) {
	metTP := met.Type()
	const wantNum = 2
	if metTP.NumIn() != wantNum {
		return nil, fmt.Errorf("method %s in %T has the wrong number of argument for a gRPC method: got %d but want %d", metTP.Name(), service, metTP.NumIn(), wantNum)
	}
	pbTP := metTP.In(1)
	itf := reflect.New(pbTP.Elem()).Interface()
	msg, ok := itf.(proto.Message)
	if !ok {
		return nil, fmt.Errorf("cannot cast %T to proto.Message", itf)
	}
	return msg, nil
}

func (s *Server) registeredService() []string {
	services := []string{}
	for k := range s.registry {
		services = append(services, k)
	}
	return services
}

func (s *Server) post(w http.ResponseWriter, r *http.Request) error {
	functionCall := r.Header.Get(_FunctionCall)
	call := strings.Split(functionCall, "/")
	if len(call) != 3 {
		return fmt.Errorf(`request.Header.%s malformed: got %q want "/service_name/method_name"`, _FunctionCall, r.Header.Get(_FunctionCall))
	}
	serviceName := call[1]
	service := s.registry[serviceName]
	if service == nil {
		return fmt.Errorf(`service %q has not been registered to the server. Registered services are: %v`, serviceName, s.registeredService())
	}
	methodName := call[2]
	method := reflect.ValueOf(service).MethodByName(methodName)
	if method.IsZero() {
		return fmt.Errorf("method %q cannot be found for service %T", methodName, service)
	}
	msg, err := protoFromMethod(service, method)
	if err != nil {
		return fmt.Errorf("cannot allocate method arguments: %v", err)
	}
	contentTypeRequest := r.Header.Get(_ContentType)
	contentTypeWant := contentType(msg)
	if contentTypeRequest != contentTypeWant {
		return fmt.Errorf("invalid content type for %s: got %q but want %q", functionCall, contentTypeRequest, contentTypeWant)
	}
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return fmt.Errorf("cannot read request body: %v", err)
	}
	defer r.Body.Close()
	if err := proto.Unmarshal(body, msg); err != nil {
		return fmt.Errorf("cannot unserialize the request: %v", err)
	}
	ctx := context.Background()
	in := []reflect.Value{
		reflect.ValueOf(ctx),
		reflect.ValueOf(msg),
	}
	out := method.Call(in)
	if len(out) != 2 {
		return fmt.Errorf("call %s returned the wrong number of values: got %v=%d but want ([proto], error)", functionCall, out, len(out))
	}
	if !out[1].IsZero() {
		return fmt.Errorf("call %s returned an error: %v", functionCall, out[1].Interface())
	}
	if out[0].IsZero() {
		return nil
	}
	obj := out[0].Interface()
	resp, ok := obj.(proto.Message)
	if !ok {
		return fmt.Errorf("call %s returned type %T which be cannot casted to proto.Message", functionCall, obj)
	}
	accept := r.Header.Get(_Accept)
	if accept != contentType(resp) {
		return fmt.Errorf("client accept %s but server will return %s", accept, contentType(resp))
	}
	respBody, err := proto.Marshal(resp)
	if err != nil {
		return fmt.Errorf("call %s returned an error: %v", functionCall, out[1].Interface())
	}
	if _, err := w.Write(respBody); err != nil {
		return fmt.Errorf("call %s cannot write protocol buffer data %T to the HTTP reponse: %v", functionCall, resp, err)
	}
	w.Header().Set(_ContentType, contentError)
	return nil
}

// Post serving httpgrpc requests.
// It needs to be registered on a http handler.
func (s *Server) Post(w http.ResponseWriter, r *http.Request) {
	if err := s.post(w, r); err != nil {
		wErr(w, err.Error())
	}
}
