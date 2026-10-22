#include "WickedEngine.h"
#include "wiGraphicsDevice_Vulkan.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

// Regression test: when the graphics device can't be created, the application must exit with an error,
//	instead of returning from the device constructor and crashing on the missing device.
//	The Vulkan device is created in a child process where no Vulkan driver can be found.

namespace
{
	constexpr int CONSTRUCTOR_RETURNED = 3;
	// What the child process writes, in a file of this run:
	const std::string child_output = "GraphicsDeviceFatalErrorTest_child_" + std::to_string(getpid()) + ".log";

	int CreateDeviceWithoutDriver()
	{
		setenv("VK_ICD_FILENAMES", "/nonexistent/vulkan_icd.json", 1);
		setenv("VK_DRIVER_FILES", "/nonexistent/vulkan_icd.json", 1);
		unsetenv("VK_ADD_DRIVER_FILES");
		// No error message box (Wayland clients also find the default display through XDG_RUNTIME_DIR):
		unsetenv("DISPLAY");
		unsetenv("WAYLAND_DISPLAY");
		unsetenv("XDG_RUNTIME_DIR");

		wi::jobsystem::Initialize();
		wi::graphics::GraphicsDevice_Vulkan* device = new wi::graphics::GraphicsDevice_Vulkan(nullptr);
		(void)device;
		return CONSTRUCTOR_RETURNED;
	}
}

int main(int argc, char** argv)
{
	if (argc > 1 && std::strcmp(argv[1], "--child") == 0)
	{
		return CreateDeviceWithoutDriver();
	}

	pid_t pid = fork();
	if (pid < 0)
	{
		std::cerr << "GraphicsDeviceFatalError test failed: fork() failed\n";
		return 1;
	}
	if (pid == 0)
	{
		// The output of the child goes to a file, which the parent checks:
		const int output = open(child_output.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
		if (output >= 0)
		{
			dup2(output, STDOUT_FILENO);
			dup2(output, STDERR_FILENO);
			close(output);
		}
		// exec instead of continuing in the fork, so that the child starts like a fresh application
		execl("/proc/self/exe", argv[0], "--child", (char*)nullptr);
		_exit(127);
	}

	int status = 0;
	int waited_ms = 0;
	while (waitpid(pid, &status, WNOHANG) == 0)
	{
		if (waited_ms > 60000)
		{
			kill(pid, SIGKILL);
			waitpid(pid, &status, 0);
			std::remove(child_output.c_str());
			std::cerr << "GraphicsDeviceFatalError test failed: device creation did not end in 60 seconds\n";
			return 1;
		}
		usleep(10000);
		waited_ms += 10;
	}

	std::string output;
	{
		std::ifstream file(child_output);
		std::stringstream text;
		text << file.rdbuf();
		output = text.str();
	}
	std::remove(child_output.c_str());
	std::cout << output;

#ifndef NDEBUG
	// In debug builds the failed Vulkan call asserts (vulkan_check()) before the device ends the process, which also ends it.
	//	Only the assert of the failed instance creation (which logs the failed call first) is accepted:
	const bool instance_failed = output.find("vkCreateInstance") != std::string::npos || output.find("volkInitialize") != std::string::npos;
	if (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT && instance_failed && output.find("Assertion") != std::string::npos)
	{
		std::cout << "GraphicsDeviceFatalError test passed (the failed instance creation asserted in this debug build)\n";
		return 0;
	}
#endif
	if (WIFSIGNALED(status))
	{
		std::cerr << "GraphicsDeviceFatalError test failed: device creation crashed with signal " << WTERMSIG(status) << '\n';
		return 1;
	}
	if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_FAILURE)
	{
		std::cerr << "GraphicsDeviceFatalError test failed: device creation ended with status " << (WIFEXITED(status) ? WEXITSTATUS(status) : -1);
		if (WIFEXITED(status) && WEXITSTATUS(status) == CONSTRUCTOR_RETURNED)
		{
			std::cerr << " (the constructor returned)";
		}
		std::cerr << ", expected EXIT_FAILURE\n";
		return 1;
	}

	std::cout << "GraphicsDeviceFatalError test passed\n";
	return 0;
}
