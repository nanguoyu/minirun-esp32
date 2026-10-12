idf_component_register(SRCS "main.c" "display.c" "pipeline.c" "../../src/lfm.c" "../../src/kv_file.c" INCLUDE_DIRS "../../src" REQUIRES fatfs sdmmc esp_driver_sdmmc esp_driver_usb_serial_jtag esp_timer mbedtls esp_lcd esp_driver_spi esp_driver_gpio esp_driver_tsens)
target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off)

# Keep strict FP accumulation; optimize code generation without fast-math.
target_compile_options(${COMPONENT_LIB} PRIVATE -O3)

set(MINIRUN_TILE_BYTES 32768 CACHE STRING "Bounded DMA tile size")
if(NOT MINIRUN_TILE_BYTES STREQUAL "32768" AND NOT MINIRUN_TILE_BYTES STREQUAL "49152")
    message(FATAL_ERROR "MINIRUN_TILE_BYTES must be 32768 or 49152")
endif()
target_compile_definitions(${COMPONENT_LIB} PRIVATE LFM_IO=${MINIRUN_TILE_BYTES})

option(MINIRUN_MODEL_ALIGNED "Use value-preserving 4096-byte GGUF layout" ON)
if(MINIRUN_MODEL_ALIGNED)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE LFM_ALIGNED_MODEL=1)
endif()

set(MINIRUN_CONTEXT 1024 CACHE STRING "Bounded context capacity")
if(NOT MINIRUN_CONTEXT MATCHES "^(64|128|256|512|1024)$")
    message(FATAL_ERROR "MINIRUN_CONTEXT must be 64, 128, 256, 512 or 1024")
endif()
target_compile_definitions(${COMPONENT_LIB} PRIVATE LFM_CONTEXT=${MINIRUN_CONTEXT})

set(MINIRUN_KV_BITS 32 CACHE STRING "KV storage precision; 16 is experimental")
if(NOT MINIRUN_KV_BITS MATCHES "^(16|32)$")
    message(FATAL_ERROR "MINIRUN_KV_BITS must be 16 or 32")
endif()
target_compile_definitions(${COMPONENT_LIB} PRIVATE LFM_KV_BITS=${MINIRUN_KV_BITS})

option(MINIRUN_KV_FILE "Store conversation cache on SD" ON)
if(MINIRUN_KV_FILE)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE LFM_KV_FILE=1)
endif()

option(MINIRUN_KV_SHARED "Share KV page reads across a causal microbatch" ON)
if(MINIRUN_KV_SHARED)
    if(NOT MINIRUN_KV_FILE)
        message(FATAL_ERROR "Shared KV reads require MINIRUN_KV_FILE")
    endif()
    target_compile_definitions(${COMPONENT_LIB} PRIVATE LFM_KV_SHARED=1)
endif()
