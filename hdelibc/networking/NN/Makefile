CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -I./utils 
LDFLAGS =

MNIST_SRCS = mnist/mnist_loader.cpp ff_neural_net.cpp utils/utils.cpp
MNIST_OBJS = $(MNIST_SRCS:.cpp=.o)

TRAIN_SRCS = mnist/train.cpp $(MNIST_SRCS)
TRAIN_OBJS = $(TRAIN_SRCS:.cpp=.o)
TRAIN_TARGET = train.out

INFERENCE_SRCS = mnist/inference.cpp $(MNIST_SRCS)
INFERENCE_OBJS = $(INFERENCE_SRCS:.cpp=.o)
INFERENCE_TARGET = inference.out

all: $(TRAIN_TARGET) $(INFERENCE_TARGET)

$(TRAIN_TARGET): $(TRAIN_OBJS)
	$(CXX) $(TRAIN_OBJS) -o $@ $(LDFLAGS)

$(INFERENCE_TARGET): $(INFERENCE_OBJS)
	$(CXX) $(INFERENCE_OBJS) -o $@ $(LDFLAGS)

mnist/%.o: mnist/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cpp
	@mkdir -p $(dir $@) 
	$(CXX) $(CXXFLAGS) -c $< -o $@

utils/%.o: utils/%.cpp 
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(MNIST_OBJS) $(TRAIN_OBJS) $(INFERENCE_OBJS) $(TRAIN_TARGET) $(INFERENCE_TARGET) weights.dat
	find mnist utils -name "*.o" -type f -delete

.PHONY: all clean