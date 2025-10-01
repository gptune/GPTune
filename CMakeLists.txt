######################################################################
#
# CMakeLists.txt for GPTune
#
######################################################################

# Required version
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

# SET(CMAKE_INSTALL_PREFIX < install_path >)

# Project version numbers
project(gptuneclcm C CXX Fortran)
set(VERSION_MAJOR "5")
set(VERSION_MINOR "0")
set(VERSION_BugFix "0")
set(PROJECT_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_BugFix})

######################################################################
#
# Load all macros:
#
FILE(GLOB _macro_files "cmake/Macros/*.cmake")
MESSAGE(STATUS "Include ${CMAKE_SOURCE_DIR}/cmake/setup_external_macros.cmake")
FOREACH(_file ${_macro_files})
  MESSAGE(STATUS "Include ${_file}")
  INCLUDE(${_file})
ENDFOREACH()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/scikit-build/skbuild/resources/cmake/")

######################################################################
#
# Usual initialization stuff
#
######################################################################
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)    ## ????
set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")

#---- For shared library

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# the RPATH to be used when installing
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#----

if (BUILD_SHARED_LIBS)
  message("-- gptuneclcm will be built as a shared library.")
else()
  message("-- gptuneclcm will be built as a static library.")
endif()

enable_language (C)
enable_language (CXX)
enable_language (Fortran)
set(CMAKE_C_STANDARD 11)
if (XSDK_ENABLE_Fortran)
  enable_language (Fortran)
  set(NOFORTRAN FALSE)
endif()
set(gptuneclcm_VERSION "${PROJECT_VERSION}")
set(gptuneclcm_REV "${PROJECT_REV}")


# if (NOT CMAKE_INSTALL_PREFIX)
#   set(CMAKE_INSTALL_PREFIX /usr/local)
# endif()


######################################################################
#
# Add compiler-specific compiler flags
#
######################################################################

set(CMAKE_C_FLAGS "-std=c++11 ${CMAKE_C_FLAGS}")
include(CheckCCompilerFlag)
check_c_compiler_flag("-qopt-matmul" qoptmatmul)
if (qoptmatmul)
set(CMAKE_C_FLAGS  "-qopt-matmul ${CMAKE_C_FLAGS}")
endif()



######################################################################
#
# Find packages
#
######################################################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${CMAKE_SOURCE_DIR}/cmake/Modules/")


#---------------------- Additional Fortran linker library ---------
SET(_fortran_libs ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
FOREACH(_lib ${_fortran_libs})
  # FIND_SYSTEM_LIBRARY(${_lib}_LIBRARY NAMES ${_lib})
  set(EXTRA_LIB "-l${_lib} ${EXTRA_LIB}")
ENDFOREACH()



# find_package(PythonExtensions REQUIRED)


#--------------------- OpenMP ---------------------
find_package(OpenMP)
## include(FindOpenMP)  # Strumpack uses this

if(OPENMP_FORTRAN_FOUND)
  set(CMAKE_Fortran_FLAGS "${OpenMP_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS}")
  set(OpenMP_Fortran_FLAGS_EXPORT "${OpenMP_Fortran_FLAGS}")
  set(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES "${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES} ${OpenMP_Fortran_FLAGS}")
endif()

if(OPENMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
  set(OpenMP_CXX_FLAGS_EXPORT "${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES} ${OpenMP_CXX_FLAGS}")
endif()






 # #--------------------- BLAS ---------------------
if(TPL_BLAS_LIBRARIES)
	set(BLAS_FOUND TRUE)
else()
	find_package(BLAS)
	if(BLAS_FOUND)
	  set(TPL_BLAS_LIBRARIES "${BLAS_LIBRARIES}" CACHE FILEPATH
		"Set from FindBLAS.cmake BLAS_LIBRARIES." FORCE)
	endif()
endif()

if(BLAS_FOUND)
    message("-- Using TPL_BLAS_LIBRARIES='${TPL_BLAS_LIBRARIES}'")
    set(BLAS_LIB ${TPL_BLAS_LIBRARIES})
    # fix up BLAS library name
    string (REPLACE ";" " " BLAS_LIB_STR "${BLAS_LIB}")
    set(BLAS_LIB_EXPORT ${BLAS_LIB_STR})
 endif()


 # #--------------------- LAPACK ---------------------
if(TPL_LAPACK_LIBRARIES)
	set(LAPACK_FOUND TRUE)
else()
	find_package(LAPACK)
	if(LAPACK_FOUND)
	  set(TPL_LAPACK_LIBRARIES "${LAPACK_LIBRARIES}" CACHE FILEPATH
		"Set from FindLAPACK.cmake LAPACK_LIBRARIES." FORCE)
	endif()
endif()

if(LAPACK_FOUND)
    message("-- Using TPL_LAPACK_LIBRARIES='${TPL_LAPACK_LIBRARIES}'")
    set(LAPACK_LIB ${TPL_LAPACK_LIBRARIES})
    # fix up LAPACK library name
    string (REPLACE ";" " " LAPACK_LIB_STR "${LAPACK_LIB}")
    set(LAPACK_LIB_EXPORT ${LAPACK_LIB_STR})
 endif()

  # #--------------------- SCALAPACK ---------------------
if(TPL_SCALAPACK_LIBRARIES)
	set(SCALAPACK_FOUND TRUE)
else()
	find_package(SCALAPACK)
	if(SCALAPACK_FOUND)
	  set(TPL_SCALAPACK_LIBRARIES "${SCALAPACK_LIBRARIES}" CACHE FILEPATH
		"Set from FindSCALAPACK.cmake SCALAPACK_LIBRARIES." FORCE)
	endif()
endif()

if(SCALAPACK_FOUND)
    message("-- Using TPL_SCALAPACK_LIBRARIES='${TPL_SCALAPACK_LIBRARIES}'")
    set(SCALAPACK_LIB ${TPL_SCALAPACK_LIBRARIES})
    # fix up LAPACK library name
    string (REPLACE ";" " " SCALAPACK_LIB_STR "${SCALAPACK_LIB}")
    set(SCALAPACK_LIB_EXPORT ${SCALAPACK_LIB_STR})
 endif()


#--------------------- MPI ---------------------
find_package(MPI)
if(MPI_C_FOUND )
    set(CMAKE_C_FLAGS "${MPI_C_COMPILE_FLAGS} ${CMAKE_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_C_LINK_FLAGS}" )
endif()
if(MPI_CXX_FOUND )
    set(CMAKE_CXX_FLAGS "${MPI_CXX_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}" )
endif()


# added a few compiler flags for Fortran
include(CheckFortranCompilerFlag)
check_fortran_compiler_flag("-ffree-line-length-none" COMPILER_GNU)
if (COMPILER_GNU)
set(CMAKE_Fortran_FLAGS "-fno-range-check -ffree-line-length-none -ffixed-line-length-none -fimplicit-none -lpthread ${CMAKE_Fortran_FLAGS}")
check_fortran_compiler_flag("-fallow-argument-mismatch" GNU10)
if (GNU10)
set(CMAKE_Fortran_FLAGS "-fallow-argument-mismatch ${CMAKE_Fortran_FLAGS}")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
set(CMAKE_Fortran_FLAGS "-fno-range-check -fbacktrace -fbounds-check -Wconversion ${CMAKE_Fortran_FLAGS}")
endif()
endif()


# python.
set(Python3_FIND_STRATEGY LOCATION) # Prioritize finding Python based on its location
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3 interpreter: ${Python3_EXECUTABLE}")
message(STATUS "Python3 installation directory: ${Python3_SITEARCH}")
set(GPTUNE_INSTALL_PATH "" CACHE STRING "GPTune module installation path")
mark_as_advanced(GPTUNE_INSTALL_PATH)


# Set python files
set(GPTUNE_PYTHON_FILES
    GPTune/__init__.py
    GPTune/__version__.py
    GPTune/callhpbandster.py
    GPTune/callopentuner.py
    GPTune/computer.py
    GPTune/database.py
    GPTune/data.py
    GPTune/gptune.py
    GPTune/lcm.py
    GPTune/model.py
    GPTune/options.py
    GPTune/problem.py
    GPTune/sample.py
    GPTune/search.py
)


# Set C source directory
set(SOURCE_DIR "gptuneclcm")
# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})
set(headers "${SOURCE_DIR}/lcm.h")
set(sources "${SOURCE_DIR}/lcm.c")
add_library(_gptuneclcm ${sources} ${HEADERS})
target_link_libraries(_gptuneclcm
                      ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${BLAS_LIB} ${LAPACK_LIB} ${SCALAPACK_LIB} m)
set_target_properties(_gptuneclcm PROPERTIES
                      VERSION ${PROJECT_VERSION} SOVERSION ${VERSION_MAJOR}
)


# Set Fortran source directory
set(SOURCE_DIR_SCALAPACK_EX "${CMAKE_SOURCE_DIR}/examples/Scalapack-PDGEQRF/scalapack-driver/src/")
include_directories(${SOURCE_DIR_SCALAPACK_EX})
#set(sources_scalapack_example "${SOURCE_DIR_SCALAPACK_EX}/pdqrdriver.f ${SOURCE_DIR_SCALAPACK_EX}/pdqrinfo.f")
set(sources_scalapack_example
${SOURCE_DIR_SCALAPACK_EX}/pdqrdriver.f
${SOURCE_DIR_SCALAPACK_EX}/pdqrinfo.f
${SOURCE_DIR_SCALAPACK_EX}/pdmatgen.f
${SOURCE_DIR_SCALAPACK_EX}/pdgeqrrv.f
${SOURCE_DIR_SCALAPACK_EX}/pdlafchk.f
${SOURCE_DIR_SCALAPACK_EX}/pdgeqlrv.f
${SOURCE_DIR_SCALAPACK_EX}/pdgerqrv.f
${SOURCE_DIR_SCALAPACK_EX}/pdtzrzrv.f
${SOURCE_DIR_SCALAPACK_EX}/pdgelqrv.f
${SOURCE_DIR_SCALAPACK_EX}/pmatgeninc.f
)
add_executable(pdqrdriver ${sources_scalapack_example})
target_link_libraries(pdqrdriver ${MPI_Fortran_LIBRARIES} ${BLAS_LIB} ${LAPACK_LIB} ${SCALAPACK_LIB} m)


# # Generate Python module
# add_subdirectory(lib/pybind11)
# pybind11_add_module(GPTune ${sources} ${headers} "${SOURCE_DIR}/bindings.cpp")

# python_extension_module(_gptuneclcm)




# Installation setup.
if(GPTUNE_INSTALL_PATH STREQUAL "")
    message(STATUS "GPTune will be installed in the default location: ${Python3_SITEARCH}")
    set(_GPTUNE_INSTALL_DIR "${Python3_SITEARCH}/GPTune")
else()
    message(STATUS "GPTune will be installed in the custom location: ${GPTUNE_INSTALL_PATH}")
    set(_GPTUNE_INSTALL_DIR "${GPTUNE_INSTALL_PATH}/GPTune")
endif()



# Define GNU standard installation directories
include(GNUInstallDirs)

install(TARGETS _gptuneclcm LIBRARY DESTINATION ${_GPTUNE_INSTALL_DIR}
  # DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(TARGETS pdqrdriver RUNTIME DESTINATION ${_GPTUNE_INSTALL_DIR}
  # DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${headers}
  DESTINATION ${_GPTUNE_INSTALL_DIR}
  # DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


# Install the Python files.
install(FILES ${GPTUNE_PYTHON_FILES} 
  DESTINATION ${_GPTUNE_INSTALL_DIR})

unset(_GPTUNE_INSTALL_DIR)
