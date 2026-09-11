cmake_minimum_required(VERSION 3.16)
project(mercan VERSION 0.1.2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MERCAN_LLAMA_DIR "${CMAKE_SOURCE_DIR}/build/_deps/llama-src" CACHE PATH "Path to patched llama.cpp source")
option(MERCAN_BUILD_SHARED "Build libmercan as a shared library" OFF)
option(MERCAN_BUILD_SDK_TESTS "Build Mercan SDK ABI regression tests" OFF)

if(NOT EXISTS "${MERCAN_LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Patched llama.cpp not found at ${MERCAN_LLAMA_DIR}. Run scripts/prepare_llama.sh first or set MERCAN_LLAMA_DIR.")
endif()

# The downloadable Mercan CLI must not require separate llama/ggml shared libraries.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

add_subdirectory("${MERCAN_LLAMA_DIR}" "${CMAKE_BINARY_DIR}/llama" EXCLUDE_FROM_ALL)

set(MERCAN_RUNTIME_SOURCES
    runtime/libmercan/src/mercan.cpp
    core/model_loader.cpp
    core/architecture_registry.cpp
    core/tokenizer_registry.cpp
    core/builtin_plugins.cpp
    architectures/nedolm/architecture.cpp
    architectures/nedolm/tokenizer.cpp
)

if(MERCAN_BUILD_SHARED)
    add_library(libmercan SHARED ${MERCAN_RUNTIME_SOURCES})
    target_compile_definitions(libmercan PRIVATE MERCAN_BUILD_SHARED=1)
else()
    add_library(libmercan STATIC ${MERCAN_RUNTIME_SOURCES})
endif()
set_target_properties(libmercan PROPERTIES OUTPUT_NAME mercan)
target_include_directories(libmercan
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime/libmercan/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        "${CMAKE_SOURCE_DIR}/core"
        "${MERCAN_LLAMA_DIR}/include"
        "${MERCAN_LLAMA_DIR}/ggml/include")
target_link_libraries(libmercan PUBLIC llama)

add_executable(mercan cli/main.cpp)
target_link_libraries(mercan PRIVATE libmercan)
if(GGML_CUDA)
    target_compile_definitions(mercan PRIVATE MERCAN_CUDA_BUILD=1)
endif()

if(MERCAN_BUILD_SDK_TESTS)
    enable_testing()
    add_executable(mercan-sdk-prefix-test tests/sdk_prefix_compat.cpp)
    target_link_libraries(mercan-sdk-prefix-test PRIVATE libmercan)
    add_test(NAME mercan-sdk-prefix-compat COMMAND mercan-sdk-prefix-test)
endif()

include(GNUInstallDirs)
install(TARGETS mercan libmercan
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
    runtime/libmercan/include/mercan.h
    runtime/libmercan/include/mercan_arch.h
    runtime/libmercan/include/mercan_tokenizer.h
    runtime/libmercan/include/mercan_graph.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
