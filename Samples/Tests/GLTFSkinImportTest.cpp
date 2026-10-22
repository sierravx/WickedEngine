#include "WickedEngine.h"
#include "ModelImporter.h"
#include "VulkanTestProcess.h"

#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

// Regression test for the skins of the editor's glTF importer: a skin without inverse bind matrices gets an identity matrix
//	for each joint (glTF 2.0 specification), instead of an assert and an armature without matrices, which the skinning of the
//	scene update reads past. The exporter writes a matrix for each joint then.
//	The import updates the scene, which needs a graphics device: the test runs with RunTestWithVulkanDevice(), and it is
//	skipped if there is none.

using namespace wi::ecs;
using namespace wi::scene;

namespace
{
	constexpr int TEST_FAILED = 2;
	const char* skins_file = "GLTFSkinImportTest_skins.gltf";
	const char* skins_bin = "GLTFSkinImportTest_skins.bin"; // the uri in the file below
	const char* output_file = "GLTFSkinImportTest_output.gltf";

	// Removes the output of the export: the file and its buffers (tinygltf names them after the file, numbered after the first one)
	void RemoveOutputFiles()
	{
		std::remove(output_file);
		std::remove("GLTFSkinImportTest_output.bin");
		for (int i = 0; i < 64; ++i)
		{
			std::remove(("GLTFSkinImportTest_output" + std::to_string(i) + ".bin").c_str());
		}
	}

	int failures = 0;

	void Check(bool condition, const std::string& message)
	{
		if (!condition)
		{
			std::cerr << "GLTFSkinImport test failed: " << message << '\n';
			failures++;
		}
	}

	bool WriteFile(const char* path, const void* data, size_t size)
	{
		FILE* file = fopen(path, "wb");
		if (file == nullptr)
			return false;
		bool written = fwrite(data, size, 1, file) == 1;
		written &= fclose(file) == 0;
		return written;
	}

	// The inverse bind matrix of the file: a scale and a translation that the Z mirror of the import keeps
	const XMFLOAT4X4 file_inverse_bind_matrix = XMFLOAT4X4(2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 2, 0, 1);

	// The buffer of the file below: a triangle, the joints and weights that skin it to two joints, and one inverse bind matrix
	struct SkinBuffer
	{
		XMFLOAT3 positions[3] = { XMFLOAT3(0, 0, 0), XMFLOAT3(1, 0, 0), XMFLOAT3(0, 1, 0) };
		uint8_t joints[3][4] = { { 0, 1, 0, 0 }, { 0, 1, 0, 0 }, { 0, 1, 0, 0 } };
		XMFLOAT4 weights[3] = { XMFLOAT4(0.5f, 0.5f, 0, 0), XMFLOAT4(0.5f, 0.5f, 0, 0), XMFLOAT4(0.5f, 0.5f, 0, 0) };
		XMFLOAT4X4 inverse_bind_matrix = file_inverse_bind_matrix;
	};
	static_assert(sizeof(SkinBuffer) == 160 && offsetof(SkinBuffer, joints) == 36 && offsetof(SkinBuffer, weights) == 48 && offsetof(SkinBuffer, inverse_bind_matrix) == 96,
		"the buffer and buffer views in the file below");

	// Two skins with two joints: one without inverse bind matrices, and one with fewer matrices than joints (an invalid file)
	const char* skins_gltf = R"({
	"asset": { "version": "2.0" },
	"scene": 0,
	"scenes": [ { "nodes": [ 0, 2, 3 ] } ],
	"nodes": [
		{ "name": "JointA", "children": [ 1 ] },
		{ "name": "JointB", "translation": [ 0, 1, 0 ] },
		{ "name": "Skinned", "mesh": 0, "skin": 0 },
		{ "name": "OtherSkinned", "mesh": 1, "skin": 1 }
	],
	"meshes": [
		{ "name": "Triangle", "primitives": [ { "attributes": { "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2 } } ] },
		{ "name": "OtherTriangle", "primitives": [ { "attributes": { "POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2 } } ] }
	],
	"skins": [
		{ "name": "NoMatrices", "joints": [ 0, 1 ] },
		{ "name": "FewMatrices", "joints": [ 0, 1 ], "inverseBindMatrices": 3 }
	],
	"buffers": [ { "byteLength": 160, "uri": "GLTFSkinImportTest_skins.bin" } ],
	"bufferViews": [
		{ "buffer": 0, "byteOffset": 0, "byteLength": 36 },
		{ "buffer": 0, "byteOffset": 36, "byteLength": 12 },
		{ "buffer": 0, "byteOffset": 48, "byteLength": 48 },
		{ "buffer": 0, "byteOffset": 96, "byteLength": 64 }
	],
	"accessors": [
		{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [ 0, 0, 0 ], "max": [ 1, 1, 0 ] },
		{ "bufferView": 1, "componentType": 5121, "count": 3, "type": "VEC4" },
		{ "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4" },
		{ "bufferView": 3, "componentType": 5126, "count": 1, "type": "MAT4" }
	]
})";

	bool SameMatrix(const XMFLOAT4X4& a, const XMFLOAT4X4& b)
	{
		for (int row = 0; row < 4; ++row)
		{
			for (int column = 0; column < 4; ++column)
			{
				if (std::abs(a.m[row][column] - b.m[row][column]) > 1e-5f)
					return false;
			}
		}
		return true;
	}

	// The armature of each skin has a matrix for each joint: the matrices of the file, then the identity
	void CheckArmatures(Scene& scene, const std::string& what)
	{
		const ArmatureComponent* no_matrices = scene.armatures.GetComponent(scene.Entity_FindByName("NoMatrices"));
		Check(no_matrices != nullptr && no_matrices->boneCollection.size() == 2 && no_matrices->inverseBindMatrices.size() == 2 &&
			SameMatrix(no_matrices->inverseBindMatrices[0], wi::math::IDENTITY_MATRIX) && SameMatrix(no_matrices->inverseBindMatrices[1], wi::math::IDENTITY_MATRIX),
			what + ": the joints of a skin without inverse bind matrices did not get the identity");
		const ArmatureComponent* few_matrices = scene.armatures.GetComponent(scene.Entity_FindByName("FewMatrices"));
		Check(few_matrices != nullptr && few_matrices->boneCollection.size() == 2 && few_matrices->inverseBindMatrices.size() == 2 &&
			SameMatrix(few_matrices->inverseBindMatrices[0], file_inverse_bind_matrix) && SameMatrix(few_matrices->inverseBindMatrices[1], wi::math::IDENTITY_MATRIX),
			what + ": the joints of a skin with fewer inverse bind matrices than joints did not get the identity after them");
	}

	int RunImport()
	{
		RemoveOutputFiles(); // the files of an earlier run must not be checked

		const SkinBuffer skin_buffer;
		const bool written = WriteFile(skins_file, skins_gltf, strlen(skins_gltf)) && WriteFile(skins_bin, &skin_buffer, sizeof(skin_buffer));
		Check(written, "the glTF file of the test could not be written");
		if (!written)
		{
			std::remove(skins_file);
			std::remove(skins_bin);
			return TEST_FAILED;
		}

		// The import updates the scene, which skins the meshes with the matrices of the armatures
		Scene scene;
		ImportModel_GLTF(skins_file, scene);
		std::remove(skins_file);
		std::remove(skins_bin);
		CheckArmatures(scene, "import");

		// The export writes a matrix for each joint, and the armatures are imported back the same way
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
		Check(model.skins.size() == 2, "export: the skins were not written");
		for (const tinygltf::Skin& skin : model.skins)
		{
			Check(skin.inverseBindMatrices >= 0 && skin.inverseBindMatrices < (int)model.accessors.size() &&
				model.accessors[skin.inverseBindMatrices].count == skin.joints.size(), "export: the skin " + skin.name + " has not a matrix for each joint");
		}
		Scene imported;
		ImportModel_GLTF(output_file, imported);
		RemoveOutputFiles();
		CheckArmatures(imported, "import of the export");

		if (failures > 0)
			return TEST_FAILED;
		std::cout << "GLTFSkinImport test passed\n";
		return 0;
	}
}

int main(int argc, char** argv)
{
	return RunTestWithVulkanDevice(argc, argv, "GLTFSkinImport", RunImport);
}
