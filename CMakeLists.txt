cmake_minimum_required(VERSION 3.13.4)

project(DifferenceStatistics)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/DifferenceStatistics")
set(EXTENSION_CATEGORY "Shape Analysis")
set(EXTENSION_CONTRIBUTORS "Kedar Madi (Virginia Tech), Jared Vicory (Kitware)")
set(EXTENSION_DESCRIPTION "Compute correlation between shape differences and covariates")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/DifferenceStatistics.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/DifferenceStatistics/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DifferenceStatistics)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
