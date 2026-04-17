CC := clang
SRC := inference.c
OUT_DYLIB := libhybrid_engine.dylib
OUT_SO := libhybrid_engine.so
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
ORT_ROOT ?= ../assets/onnxruntime-osx-arm64-1.24.4
NATIVE_OUT := $(OUT_DYLIB)
NATIVE_LDFLAGS := -dynamiclib -Wl,-rpath,@loader_path/../assets/onnxruntime-osx-arm64-1.24.4/lib
SO_LDFLAGS := -shared -Wl,-rpath,@loader_path/../assets/onnxruntime-osx-arm64-1.24.4/lib
else
ORT_ROOT ?= ../assets/onnxruntime-linux-x64
NATIVE_OUT := $(OUT_SO)
NATIVE_LDFLAGS := -shared -Wl,-rpath,'$$ORIGIN/../assets/onnxruntime-linux-x64/lib'
SO_LDFLAGS := $(NATIVE_LDFLAGS)
endif

ORT_INC := $(ORT_ROOT)/include
ORT_LIB := $(ORT_ROOT)/lib

CFLAGS := -fPIC -I$(ORT_INC)
LINK_LIBS := -L$(ORT_LIB) -lonnxruntime

.PHONY: all build so clean

all: build

build: $(NATIVE_OUT)

$(OUT_DYLIB): $(SRC)
	$(CC) $(CFLAGS) $(NATIVE_LDFLAGS) -o $@ $< $(LINK_LIBS)

so: $(SRC)
	$(CC) $(CFLAGS) $(SO_LDFLAGS) -o $(OUT_SO) $< $(LINK_LIBS)

clean:
	rm -f $(OUT_DYLIB) $(OUT_SO)
