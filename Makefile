# graph-gpu: BFS + PageRank on Apple Silicon (Metal).
# Plain Makefile on purpose (no CMake): Apple clang + Command Line Tools only.

CXX      := clang++
LIBOMP   := $(shell brew --prefix libomp 2>/dev/null)
BUILD    := build
OBJDIR   := $(BUILD)/obj

# Homebrew's libomp is built for the running macOS (26.x), while the Command
# Line Tools SDK is 13.3. Linking against it with the SDK's default deployment
# target produces an ld warning on every link, so we target the OS we run on.
MACOS_MIN := -mmacosx-version-min=26.0

# Shaders are compiled at runtime (no offline `metal` compiler without Xcode),
# so binaries need to know where src/shaders lives. Override with the
# GGPU_SHADER_DIR environment variable if the tree is moved.
SHADER_DIR := $(CURDIR)/src/shaders

CPPFLAGS := -Isrc -I$(LIBOMP)/include -DGGPU_SHADER_DIR='"$(SHADER_DIR)"' -MMD -MP
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -march=native $(MACOS_MIN) -Xpreprocessor -fopenmp
OBJCXXFLAGS := $(CXXFLAGS) -fobjc-arc
LDFLAGS  := $(MACOS_MIN) -L$(LIBOMP)/lib
LDLIBS   := -lomp -framework Metal -framework Foundation -lz

# Code shared by every executable. Grows phase by phase.
LIB_SRCS := src/metal/context.mm \
            src/graph/csr.cpp src/graph/snap.cpp src/graph/transform.cpp \
            src/graph/generators.cpp src/cpu/bfs_cpu.cpp
LIB_OBJS := $(patsubst src/%,$(OBJDIR)/%.o,$(LIB_SRCS))

BINS := $(BUILD)/gmicro $(BUILD)/gtest $(BUILD)/gconvert $(BUILD)/gstats

.PHONY: all test clean
all: $(BINS)

$(BUILD)/gmicro: $(OBJDIR)/bench/micro.mm.o $(LIB_OBJS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/gconvert: $(OBJDIR)/graph/convert.cpp.o $(LIB_OBJS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/gstats: $(OBJDIR)/graph/stats.cpp.o $(LIB_OBJS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD)/gtest: $(OBJDIR)/tests/test_main.cpp.o $(LIB_OBJS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(OBJDIR)/%.cpp.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.mm.o: src/%.mm
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(OBJCXXFLAGS) -c $< -o $@

$(OBJDIR)/tests/%.cpp.o: tests/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(OBJCXXFLAGS) -x objective-c++ -c $< -o $@

test: $(BUILD)/gtest
	./$(BUILD)/gtest

clean:
	rm -rf $(BUILD)

-include $(shell find $(OBJDIR) -name '*.d' 2>/dev/null)
