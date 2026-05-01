include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(RADE_CORE_SOURCES
    rade_enc.c
    rade_dec.c
    rade_enc_data.c
    rade_dec_data.c
    complex_bpf.c
    frame_sync.c
    frame_sync_data.c
    rx2_frontend.c
    rx2_coarse_sync.c
    rx2_demod.c
    rx2_eoo.c
    rx2_frame_sync.c
    rx2_receiver.c
    rx2_model_data.c
    tx2_encode.c
    tx2_model_data.c)

set(RADE_PURE_C_API_SOURCES
    rade_api_pure_c.c
    ${RADE_CORE_SOURCES})

add_executable(ch ch.c ch_fdmdv.c)
target_link_libraries(ch m)

add_executable(mksine mksine.c)
target_link_libraries(mksine m)

add_executable(lpcnet_demo lpcnet_demo.c)
target_link_libraries(lpcnet_demo opus m)

add_library(rade_pure_c STATIC ${RADE_PURE_C_API_SOURCES})
target_link_libraries(rade_pure_c PUBLIC opus m)
target_compile_definitions(rade_pure_c PRIVATE -DIS_BUILDING_RADE_API=1)

if(RADAE_BUILD_LEGACY_PYTHON_API)
add_library(rade
    rade_api.c
    ${RADE_CORE_SOURCES})
if(APPLE OR WIN32)
target_link_libraries(rade Python3::Python opus)
else(APPLE OR WIN32)
target_link_libraries(rade Python3::Python)
endif(APPLE OR WIN32)
target_include_directories(rade PRIVATE
    "$<TARGET_PROPERTY:Python3::NumPy,INTERFACE_INCLUDE_DIRECTORIES>")
target_compile_definitions(rade PRIVATE -DIS_BUILDING_RADE_API=1)
set_target_properties(rade PROPERTIES
    SOVERSION "0.1"
    PUBLIC_HEADER "rade_api.h"
)

add_executable(radae_tx radae_tx.c)
target_link_libraries(radae_tx rade opus m Python3::Python)
target_include_directories(radae_tx PRIVATE
    "$<TARGET_PROPERTY:Python3::NumPy,INTERFACE_INCLUDE_DIRECTORIES>")

add_executable(radae_rx radae_rx.c)
target_link_libraries(radae_rx rade opus m Python3::Python)
target_include_directories(radae_rx PRIVATE
    "$<TARGET_PROPERTY:Python3::NumPy,INTERFACE_INCLUDE_DIRECTORIES>")
endif()

add_executable(radae_rx_v2 radae_rx_v2.c)
target_link_libraries(radae_rx_v2 rade_pure_c)

add_executable(radae_tx_v2 radae_tx_v2.c)
target_link_libraries(radae_tx_v2 rade_pure_c)

if (NOT WIN32)
if(RADAE_BUILD_LEGACY_PYTHON_API)
add_executable(test_rade_enc test_rade_enc.c)
target_link_libraries(test_rade_enc rade opus m)

add_executable(test_rade_dec test_rade_dec.c)
target_link_libraries(test_rade_dec rade opus m)
endif()

add_executable(test_frame_sync test_frame_sync.c frame_sync.c frame_sync_data.c)
target_link_libraries(test_frame_sync m)

add_executable(test_complex_bpf test_complex_bpf.c complex_bpf.c)
target_link_libraries(test_complex_bpf m)

add_executable(test_rx2_frontend test_rx2_frontend.c rx2_frontend.c)
target_link_libraries(test_rx2_frontend m)

add_executable(test_rx2_coarse_sync test_rx2_coarse_sync.c rx2_coarse_sync.c)
target_link_libraries(test_rx2_coarse_sync m)

add_executable(test_rx2_demod test_rx2_demod.c rx2_demod.c)
target_link_libraries(test_rx2_demod m)

add_executable(test_rx2_eoo test_rx2_eoo.c rx2_eoo.c)
target_link_libraries(test_rx2_eoo m)

add_executable(test_rx2_frame_sync test_rx2_frame_sync.c rx2_frame_sync.c frame_sync.c frame_sync_data.c rade_dec.c rade_dec_data.c)
target_link_libraries(test_rx2_frame_sync opus m)

add_executable(test_rx2_receiver
    test_rx2_receiver.c
    rx2_receiver.c
    rx2_frontend.c
    rx2_coarse_sync.c
    rx2_demod.c
    rx2_eoo.c
    rx2_frame_sync.c
    frame_sync.c
    frame_sync_data.c
    rade_dec.c
    rade_dec_data.c)
target_link_libraries(test_rx2_receiver opus m)

add_executable(test_rade_rx_v2 test_rade_rx_v2.c)
target_link_libraries(test_rade_rx_v2 rade_pure_c)

add_executable(test_tx2_encode
    test_tx2_encode.c
    tx2_encode.c
    tx2_model_data.c
    complex_bpf.c
    rade_enc.c
    rade_enc_data.c)
target_link_libraries(test_tx2_encode opus m)
endif (NOT WIN32)

if(RADAE_BUILD_LEGACY_PYTHON_API)
add_executable(write_rade_weights write_rade_weights.c)
target_link_libraries(write_rade_weights rade opus m)
endif()

if(APPLE AND BUILD_OSX_UNIVERSAL)
add_dependencies(rade_pure_c build_opus_x86)
add_dependencies(rade_pure_c build_opus_arm)
if(RADAE_BUILD_LEGACY_PYTHON_API)
add_dependencies(rade build_opus_x86)
add_dependencies(rade build_opus_arm)
endif()
else(APPLE AND BUILD_OSX_UNIVERSAL)
add_dependencies(rade_pure_c build_opus)
if(RADAE_BUILD_LEGACY_PYTHON_API)
add_dependencies(rade build_opus)
endif()
endif(APPLE AND BUILD_OSX_UNIVERSAL)
