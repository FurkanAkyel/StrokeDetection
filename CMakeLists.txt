cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(StrokeDetection)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_NAME "StrokeDetection")  # Uzantı adı
set(EXTENSION_HOMEPAGE "https://github.com/FurkanAkyel/StrokeDetection")  # GitHub veya web sayfası URL'si
set(EXTENSION_CATEGORY "Examples")  # Kategori
set(EXTENSION_CONTRIBUTORS "Furkan (Inonu University)")
set(EXTENSION_DESCRIPTION "This is a classification and segmentation module for brain stroke.")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(StrokeDetectionModule)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})