#include "WickedEngine.h"
#include "wiGraphicsDevice_Vulkan.h"
#include "ModelImporter.h"

#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

// Regression test for the cameras of the editor's glTF importer: every camera node gets the parameters of the glTF camera
//	that it refers to, whatever the order of the nodes and the cameras, with orthographic cameras, cameras that no node uses
//	and cameras that several nodes use. The exporter refers to the cameras the same way, so that they survive a round trip.
//	The import updates the scene, which needs a graphics device: the test runs in a child process that creates a Vulkan
//	device (like Mesa's software lavapipe), and it is skipped if there is none.

using namespace wi::ecs;
using namespace wi::scene;

namespace
{
	constexpr int TEST_SKIPPED = 77;
	constexpr int TEST_FAILED = 2;
	const char* cameras_file = "GLTFCameraImportTest_cameras.gltf";
	const char* unused_file = "GLTFCameraImportTest_unused.gltf";
	const char* output_file = "GLTFCameraImportTest_output.gltf";

	// Removes the output of the export: the file and its buffers (tinygltf names them after the file, numbered after the first one)
	void RemoveOutputFiles()
	{
		std::remove(output_file);
		std::remove("GLTFCameraImportTest_output.bin");
		for (int i = 0; i < 64; ++i)
		{
			std::remove(("GLTFCameraImportTest_output" + std::to_string(i) + ".bin").c_str());
		}
	}

	int failures = 0;

	void Check(bool condition, const std::string& message)
	{
		if (!condition)
		{
			std::cerr << "GLTFCameraImport test failed: " << message << '\n';
			failures++;
		}
	}

	bool Near(double a, double b)
	{
		return std::abs(a - b) <= 1e-4 * std::max(1.0, std::abs(b));
	}

	bool WriteFile(const char* path, const char* text)
	{
		FILE* file = fopen(path, "w");
		if (file == nullptr)
			return false;
		bool written = fputs(text, file) >= 0;
		written &= fclose(file) == 0;
		return written;
	}

	// Two perspective cameras that the nodes use in the other order than they are defined, an orthographic camera before
	//	them, a camera that no node uses, a camera that two nodes use, and a camera without zfar (an infinite projection)
	const char* cameras_gltf = R"({
	"asset": { "version": "2.0" },
	"scene": 0,
	"scenes": [ { "nodes": [ 0, 1, 2, 3, 4, 5 ] } ],
	"nodes": [
		{ "name": "FarNode", "camera": 2 },
		{ "name": "NearNode", "camera": 1 },
		{ "name": "OrthoNode", "camera": 0 },
		{ "name": "SharedNode", "camera": 4 },
		{ "name": "OtherSharedNode", "camera": 4 },
		{ "name": "InfiniteNode", "camera": 5 }
	],
	"cameras": [
		{ "name": "Ortho", "type": "orthographic", "orthographic": { "xmag": 3, "ymag": 2, "znear": 0.5, "zfar": 50 } },
		{ "name": "Near", "type": "perspective", "perspective": { "aspectRatio": 2, "yfov": 0.5, "znear": 0.2, "zfar": 20 } },
		{ "name": "Far", "type": "perspective", "perspective": { "aspectRatio": 1, "yfov": 1.2, "znear": 1, "zfar": 300 } },
		{ "name": "Unused", "type": "perspective", "perspective": { "aspectRatio": 0.5, "yfov": 0.9, "znear": 0.3, "zfar": 30 } },
		{ "name": "Shared", "type": "perspective", "perspective": { "aspectRatio": 1.5, "yfov": 0.7, "znear": 0.4, "zfar": 40 } },
		{ "name": "Infinite", "type": "perspective", "perspective": { "yfov": 0.8, "znear": 0.25 } }
	]
})";

	// More cameras than camera nodes: the unused camera comes first
	const char* unused_gltf = R"({
	"asset": { "version": "2.0" },
	"scene": 0,
	"scenes": [ { "nodes": [ 0 ] } ],
	"nodes": [ { "name": "OnlyNode", "camera": 1 } ],
	"cameras": [
		{ "name": "Unused", "type": "perspective", "perspective": { "aspectRatio": 0.5, "yfov": 0.9, "znear": 0.3, "zfar": 30 } },
		{ "name": "Used", "type": "perspective", "perspective": { "aspectRatio": 2.5, "yfov": 1.1, "znear": 0.6, "zfar": 60 } }
	]
})";

	struct ExpectedCamera
	{
		std::string node;
		bool ortho;
		float aspect; // width / height
		float fov; // of a perspective camera
		float ortho_vertical_size; // of an orthographic camera
		float znear;
		float zfar;
	};

	void CheckCameras(Scene& scene, const wi::vector<ExpectedCamera>& expected_cameras, const std::string& what)
	{
		Check(scene.cameras.GetCount() == expected_cameras.size(), what + ": not one camera component for each camera node");
		for (const ExpectedCamera& expected : expected_cameras)
		{
			const CameraComponent* camera = scene.cameras.GetComponent(scene.Entity_FindByName(expected.node));
			if (camera == nullptr)
			{
				Check(false, what + ": the node " + expected.node + " has no camera");
				continue;
			}
			bool same = camera->IsOrtho() == expected.ortho && camera->height != 0 && Near(camera->width / camera->height, expected.aspect) &&
				Near(camera->zNearP, expected.znear) && Near(camera->zFarP, expected.zfar);
			same &= expected.ortho ? Near(camera->ortho_vertical_size, expected.ortho_vertical_size) : Near(camera->fov, expected.fov);
			Check(same, what + ": the node " + expected.node + " did not get the parameters of its camera");
		}
	}

	int RunImport(int device_created_pipe)
	{
		wi::jobsystem::Initialize();
		wi::graphics::GetDevice() = new wi::graphics::GraphicsDevice_Vulkan(nullptr); // ends the process with EXIT_FAILURE without a device
		// Tell the parent that the device was created, so that later failures are not taken for a missing device:
		if (write(device_created_pipe, "1", 1) != 1)
			return TEST_FAILED;
		close(device_created_pipe);

		RemoveOutputFiles(); // the files of an earlier run must not be checked

		// The camera that a node gets before the import fills it, the camera without zfar keeps its far plane
		Scene created;
		const CameraComponent& created_camera = *created.cameras.GetComponent(created.Entity_CreateCamera("", 16, 9));

		const wi::vector<ExpectedCamera> expected_cameras = {
			{ "FarNode", false, 1, 1.2f, 0, 1, 300 },
			{ "NearNode", false, 2, 0.5f, 0, 0.2f, 20 },
			{ "OrthoNode", true, 1.5f, 0, 4, 0.5f, 50 }, // the whole height of the view, glTF has half of it
			{ "SharedNode", false, 1.5f, 0.7f, 0, 0.4f, 40 },
			{ "OtherSharedNode", false, 1.5f, 0.7f, 0, 0.4f, 40 },
			{ "InfiniteNode", false, created_camera.width / created_camera.height, 0.8f, 0, 0.25f, created_camera.zFarP },
		};

		const bool written = WriteFile(cameras_file, cameras_gltf) && WriteFile(unused_file, unused_gltf);
		Check(written, "the glTF files of the test could not be written");
		if (!written)
		{
			std::remove(cameras_file);
			std::remove(unused_file);
			return TEST_FAILED;
		}

		Scene scene;
		ImportModel_GLTF(cameras_file, scene);
		std::remove(cameras_file);
		CheckCameras(scene, expected_cameras, "import");

		// The import used to fill the camera components in the order of the glTF cameras, writing past the last component
		//	when there were more cameras than camera nodes
		Scene unused;
		ImportModel_GLTF(unused_file, unused);
		std::remove(unused_file);
		CheckCameras(unused, { { "OnlyNode", false, 2.5f, 1.1f, 0, 0.6f, 60 } }, "import with an unused camera");

		// The exported nodes refer to the cameras with their parameters, and the cameras are imported back the same way
		ExportModel_GLTF(output_file, scene);
		tinygltf::TinyGLTF loader;
		tinygltf::Model model;
		std::string err, warn;
		if (!loader.LoadASCIIFromFile(&model, &err, &warn, output_file))
		{
			Check(false, "the exported file could not be loaded: " + err);
			RemoveOutputFiles();
			return TEST_FAILED;
		}
		for (const ExpectedCamera& expected : expected_cameras)
		{
			const tinygltf::Camera* camera = nullptr;
			for (const tinygltf::Node& node : model.nodes)
			{
				if (node.name == expected.node && node.camera >= 0 && node.camera < (int)model.cameras.size())
				{
					camera = &model.cameras[node.camera];
				}
			}
			bool same = camera != nullptr;
			if (same && expected.ortho)
			{
				same = camera->type == "orthographic" && Near(camera->orthographic.ymag, expected.ortho_vertical_size * 0.5) &&
					Near(camera->orthographic.xmag, expected.ortho_vertical_size * 0.5 * expected.aspect) &&
					Near(camera->orthographic.znear, expected.znear) && Near(camera->orthographic.zfar, expected.zfar);
			}
			else if (same)
			{
				same = camera->type == "perspective" && Near(camera->perspective.aspectRatio, expected.aspect) && Near(camera->perspective.yfov, expected.fov) &&
					Near(camera->perspective.znear, expected.znear) && Near(camera->perspective.zfar, expected.zfar);
			}
			Check(same, "export: the node " + expected.node + " does not refer to a camera with its parameters");
		}
		Scene imported;
		ImportModel_GLTF(output_file, imported);
		RemoveOutputFiles();
		CheckCameras(imported, expected_cameras, "import of the export");

		if (failures > 0)
			return TEST_FAILED;
		std::cout << "GLTFCameraImport test passed\n";
		return 0;
	}
}

int main(int argc, char** argv)
{
	if (argc > 2 && std::strcmp(argv[1], "--child") == 0)
	{
		// No error message box if the device can't be created:
		unsetenv("DISPLAY");
		unsetenv("WAYLAND_DISPLAY");
		unsetenv("XDG_RUNTIME_DIR");
		const int result = RunImport(std::atoi(argv[2]));
		std::cout.flush();
		std::cerr.flush();
		std::_Exit(result); // without destroying the device in the static destructors
	}

	int device_created[2];
	if (pipe(device_created) != 0)
	{
		std::cerr << "GLTFCameraImport test failed: pipe() failed\n";
		return 1;
	}
	pid_t pid = fork();
	if (pid < 0)
	{
		std::cerr << "GLTFCameraImport test failed: fork() failed\n";
		return 1;
	}
	if (pid == 0)
	{
		close(device_created[0]);
		const std::string fd = std::to_string(device_created[1]);
		execl("/proc/self/exe", argv[0], "--child", fd.c_str(), (char*)nullptr);
		_exit(127);
	}
	close(device_created[1]);
	int status = 0;
	pid_t waited = -1;
	do
	{
		waited = waitpid(pid, &status, 0);
	} while (waited < 0 && errno == EINTR);
	if (waited != pid)
	{
		std::cerr << "GLTFCameraImport test failed: waitpid() failed\n";
		return 1;
	}
	char byte = 0;
	const bool created = read(device_created[0], &byte, 1) == 1;
	close(device_created[0]);

	if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
		return 0;
	if (!created && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE)
	{
		std::cout << "GLTFCameraImport test skipped: no Vulkan device could be created\n";
		return TEST_SKIPPED;
	}
	if (WIFSIGNALED(status))
	{
		std::cerr << "GLTFCameraImport test failed: the test process (import or export) crashed with signal " << WTERMSIG(status) << '\n';
	}
	else if (WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE)
	{
		std::cerr << "GLTFCameraImport test failed: the test process (import or export) ended with EXIT_FAILURE\n";
	}
	return 1;
}
