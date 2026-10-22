#pragma once
#include "WickedEngine.h"
#include "wiGraphicsDevice_Vulkan.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

// Runs a test that needs a graphics device (like the editor's glTF importer, which updates the scene) in a child process
//	that creates a Vulkan device (like Mesa's software lavapipe), and skips the test if there is none. Call it from main()
//	with the test, which returns 0 when it passed. The result is the exit code of the test: 0 passed, 77 skipped, else failed.
inline int RunTestWithVulkanDevice(int argc, char** argv, const std::string& name, int (*test)())
{
	constexpr int TEST_SKIPPED = 77;

	if (argc > 2 && std::strcmp(argv[1], "--child") == 0)
	{
		// No error message box if the device can't be created:
		unsetenv("DISPLAY");
		unsetenv("WAYLAND_DISPLAY");
		unsetenv("XDG_RUNTIME_DIR");
		wi::jobsystem::Initialize();
		wi::graphics::GetDevice() = new wi::graphics::GraphicsDevice_Vulkan(nullptr); // ends the process with EXIT_FAILURE without a device
		// Tell the parent that the device was created, so that later failures are not taken for a missing device:
		const int device_created_pipe = std::atoi(argv[2]);
		const bool told = write(device_created_pipe, "1", 1) == 1;
		close(device_created_pipe);
		const int result = told ? test() : 2;
		std::cout.flush();
		std::cerr.flush();
		std::_Exit(result); // without destroying the device in the static destructors
	}

	int device_created[2];
	if (pipe(device_created) != 0)
	{
		std::cerr << name << " test failed: pipe() failed\n";
		return 1;
	}
	pid_t pid = fork();
	if (pid < 0)
	{
		std::cerr << name << " test failed: fork() failed\n";
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
		std::cerr << name << " test failed: waitpid() failed\n";
		return 1;
	}
	char byte = 0;
	const bool created = read(device_created[0], &byte, 1) == 1;
	close(device_created[0]);

	if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
		return 0;
	if (!created && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE)
	{
		std::cout << name << " test skipped: no Vulkan device could be created\n";
		return TEST_SKIPPED;
	}
	if (WIFSIGNALED(status))
	{
		std::cerr << name << " test failed: the test process crashed with signal " << WTERMSIG(status) << '\n';
	}
	else if (WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE)
	{
		std::cerr << name << " test failed: the test process ended with EXIT_FAILURE\n";
	}
	return 1;
}
