CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20
LDFLAGS =

SRCS = Servers/server.cpp Servers/TestServer.cpp Servers/SimpleServer.cpp \
	   Sockets/SimpleSocket.cpp Sockets/BindingSocket.cpp Sockets/ListeningSocket.cpp \
	   Database/Database.cpp

OBJS = $(SRCS:.cpp=.o)

TARGET = server.exe

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean