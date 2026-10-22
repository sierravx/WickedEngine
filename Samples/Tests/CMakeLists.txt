cmake_minimum_required(VERSION 3.19)
set (SOURCE_FILES
	Tests.cpp
	Tests.h
)

if (WIN32)
	list (APPEND SOURCE_FILES
		main_Windows.cpp
		main_Windows.h
		Tests.rc
	)

	add_executable(Tests WIN32 ${SOURCE_FILES})

	target_link_libraries(Tests PUBLIC
		WickedEngine_Windows
	)

	set(LIB_DXCOMPILER "dxcompiler.dll")
else()
	list (APPEND SOURCE_FILES
		main_${PLATFORM}.cpp
	)

	add_executable(Tests ${SOURCE_FILES})

	target_link_libraries(Tests PUBLIC
		WickedEngine
	)

	set(LIB_DXCOMPILER "libdxcompiler.so")
endif ()

if(WICKED_ENABLE_IPO)
set_target_properties(Tests PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION ON
    INTERPROCEDURAL_OPTIMIZATION_DEBUG OFF
)
endif()

target_precompile_headers(Tests PRIVATE stdafx.h)

add_executable(CustomShaderLifecycleTest
	CustomShaderLifecycleTest.cpp
)
target_link_libraries(CustomShaderLifecycleTest PRIVATE WickedEngine)
add_test(NAME CustomShaderLifecycleTest COMMAND CustomShaderLifecycleTest)
set_target_properties(CustomShaderLifecycleTest PROPERTIES FOLDER "Tests")

add_executable(TransformDecomposeTest
	TransformDecomposeTest.cpp
)
target_link_libraries(TransformDecomposeTest PRIVATE WickedEngine)
add_test(NAME TransformDecomposeTest COMMAND TransformDecomposeTest)
set_target_properties(TransformDecomposeTest PROPERTIES FOLDER "Tests")

add_executable(FlipZAxisTransformsTest
	FlipZAxisTransformsTest.cpp
)
target_link_libraries(FlipZAxisTransformsTest PRIVATE WickedEngine)
add_test(NAME FlipZAxisTransformsTest COMMAND FlipZAxisTransformsTest)
set_target_properties(FlipZAxisTransformsTest PROPERTIES FOLDER "Tests")

add_executable(AnimationRetargetTest
	AnimationRetargetTest.cpp
)
target_link_libraries(AnimationRetargetTest PRIVATE WickedEngine)
add_test(NAME AnimationRetargetTest COMMAND AnimationRetargetTest)
set_target_properties(AnimationRetargetTest PROPERTIES FOLDER "Tests")

add_executable(RetargetAnimationTest
	RetargetAnimationTest.cpp
)
target_link_libraries(RetargetAnimationTest PRIVATE WickedEngine)
add_test(NAME RetargetAnimationTest COMMAND RetargetAnimationTest)
set_target_properties(RetargetAnimationTest PROPERTIES FOLDER "Tests")

add_executable(CubicSplineAnimationTest
	CubicSplineAnimationTest.cpp
)
target_link_libraries(CubicSplineAnimationTest PRIVATE WickedEngine)
add_test(NAME CubicSplineAnimationTest COMMAND CubicSplineAnimationTest)
set_target_properties(CubicSplineAnimationTest PROPERTIES FOLDER "Tests")

# Linux only: elsewhere the error message box of the failed device creation is modal
if (UNIX AND NOT APPLE)
	add_executable(GraphicsDeviceFatalErrorTest
		GraphicsDeviceFatalErrorTest.cpp
	)
	target_link_libraries(GraphicsDeviceFatalErrorTest PRIVATE WickedEngine)
	add_test(NAME GraphicsDeviceFatalErrorTest COMMAND GraphicsDeviceFatalErrorTest)
	set_target_properties(GraphicsDeviceFatalErrorTest PROPERTIES FOLDER "Tests")

	# The editor's glTF importer and exporter, it needs a Vulkan device (like Mesa's lavapipe) and is skipped without one
	add_executable(GLTFExportTest
		GLTFExportTest.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/../../Editor/ModelImporter_GLTF.cpp
	)
	target_include_directories(GLTFExportTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../Editor)
	target_link_libraries(GLTFExportTest PRIVATE WickedEngine)
	add_test(NAME GLTFExportTest COMMAND GLTFExportTest)
	set_tests_properties(GLTFExportTest PROPERTIES SKIP_RETURN_CODE 77)
	set_target_properties(GLTFExportTest PROPERTIES FOLDER "Tests")

	# The cameras of the editor's glTF importer, it needs a Vulkan device too and is skipped without one
	add_executable(GLTFCameraImportTest
		GLTFCameraImportTest.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/../../Editor/ModelImporter_GLTF.cpp
	)
	target_include_directories(GLTFCameraImportTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../Editor)
	target_link_libraries(GLTFCameraImportTest PRIVATE WickedEngine)
	add_test(NAME GLTFCameraImportTest COMMAND GLTFCameraImportTest)
	set_tests_properties(GLTFCameraImportTest PROPERTIES SKIP_RETURN_CODE 77)
	set_target_properties(GLTFCameraImportTest PROPERTIES FOLDER "Tests")

	# The skins of the editor's glTF importer, it needs a Vulkan device too and is skipped without one
	add_executable(GLTFSkinImportTest
		GLTFSkinImportTest.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/../../Editor/ModelImporter_GLTF.cpp
	)
	target_include_directories(GLTFSkinImportTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../Editor)
	target_link_libraries(GLTFSkinImportTest PRIVATE WickedEngine)
	add_test(NAME GLTFSkinImportTest COMMAND GLTFSkinImportTest)
	set_tests_properties(GLTFSkinImportTest PROPERTIES SKIP_RETURN_CODE 77)
	set_target_properties(GLTFSkinImportTest PROPERTIES FOLDER "Tests")

	# The STEP and NEAREST animation samplers: sampling, glTF import and export (the sampling needs a Vulkan device too)
	add_executable(GLTFStepAnimationTest
		GLTFStepAnimationTest.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/../../Editor/ModelImporter_GLTF.cpp
	)
	target_include_directories(GLTFStepAnimationTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../Editor)
	target_link_libraries(GLTFStepAnimationTest PRIVATE WickedEngine)
	add_test(NAME GLTFStepAnimationTest COMMAND GLTFStepAnimationTest)
	set_tests_properties(GLTFStepAnimationTest PROPERTIES SKIP_RETURN_CODE 77)
	set_target_properties(GLTFStepAnimationTest PROPERTIES FOLDER "Tests")
endif()


if (MSVC)
	set_property(TARGET Tests PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif ()

# Copy content to build folder:
add_dependencies(Tests Content)

add_custom_command(
	TARGET Tests POST_BUILD
	# Copy the shader library next to the executable
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${WICKED_ROOT_DIR}/WickedEngine/${LIB_DXCOMPILER} ${CMAKE_CURRENT_BINARY_DIR}
	# Copy the resources
	COMMAND ${CMAKE_COMMAND} -E ${COPY_OR_SYMLINK_DIR_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/images ${CMAKE_CURRENT_BINARY_DIR}/images
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/test_script.lua ${CMAKE_CURRENT_BINARY_DIR}/test_script.lua
	# Copy font test resources
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/yumin.ttf ${CMAKE_CURRENT_BINARY_DIR}/

	COMMENT "$<IF:$<BOOL:${WICKED_ENABLE_SYMLINKS}>,Symlinking,Copying> Tests resources"
	VERBATIM
)
