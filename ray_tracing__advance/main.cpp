/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


// ImGui - standalone example application for Glfw + Vulkan, using programmable
// pipeline If you are new to ImGui, see examples/README.txt and documentation
// at the top of imgui.cpp.

#include <Windows.h>

#include <array>

#define IMGUI_DEFINE_MATH_OPERATORS
#include "backends/imgui_impl_glfw.h"
#include "imgui.h"

#include "hello_vulkan.h"
#include "imgui/imgui_camera_widget.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvpsystem.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"

// Point cloud library
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <Eigen/Dense>
#include <random>

// CUDA includes
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>
#include <mma.h>

//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;

// Windows high performance timer
LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
LARGE_INTEGER Frequency;


// GLFW Callback functions
static void onErrorCallback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Extra UI
void renderUI(HelloVulkan& helloVk)
{
  bool changed = false;

  changed |= ImGuiH::CameraWidget();
  if(ImGui::CollapsingHeader("Light"))
  {
    auto& pc = helloVk.m_pcRaster;

    changed |= ImGui::RadioButton("Point", &pc.lightType, 0);
    ImGui::SameLine();
    changed |= ImGui::RadioButton("Spot", &pc.lightType, 1);
    ImGui::SameLine();
    changed |= ImGui::RadioButton("Infinite", &pc.lightType, 2);


    if(pc.lightType < 2)
    {
      changed |= ImGui::SliderFloat3("Light Position", &pc.lightPosition.x, -20.f, 20.f);
    }
    if(pc.lightType > 0)
    {
      changed |= ImGui::SliderFloat3("Light Direction", &pc.lightDirection.x, -1.f, 1.f);
    }
    if(pc.lightType < 2)
    {
      changed |= ImGui::SliderFloat("Light Intensity", &pc.lightIntensity, 0.f, 500.f);
    }
    if(pc.lightType == 1)
    {
      float dCutoff    = rad2deg(acos(pc.lightSpotCutoff));
      float dOutCutoff = rad2deg(acos(pc.lightSpotOuterCutoff));
      changed |= ImGui::SliderFloat("Cutoff", &dCutoff, 0.f, 45.f);
      changed |= ImGui::SliderFloat("OutCutoff", &dOutCutoff, 0.f, 45.f);
      dCutoff = dCutoff > dOutCutoff ? dOutCutoff : dCutoff;

      pc.lightSpotCutoff      = cos(deg2rad(dCutoff));
      pc.lightSpotOuterCutoff = cos(deg2rad(dOutCutoff));
    }
  }

  changed |= ImGui::SliderInt("Max Frames", &helloVk.m_maxFrames, 1, 1000);
  if(changed)
    helloVk.resetFrame();
}

pcl::PointCloud<pcl::PointXYZRGB> read_PLY()
{
  pcl::PointCloud<pcl::PointXYZRGB> mesh;

  pcl::io::loadPLYFile(
      "C:/Users/jaido/OneDrive - UW/GPU-Accelerated-Visualization/GPU-Accelerated-Visualization/PointCloudRendering_RT/data/porch.ply",
      mesh);

  return mesh;
}

// A function that takes a pointer to a c array, the number of rows and columns, and returns an Eigen matrix
Eigen::MatrixXd init_matrix(double* array, int rows, int cols)
{
  // Create an Eigen matrix with the same size as the array
  Eigen::MatrixXd mat(rows, cols);
  // Copy the data from the array to the matrix
  for(int i = 0; i < rows; i++)
  {
    for(int j = 0; j < cols; j++)
    {
      mat(i, j) = array[i * cols + j];
    }
  }
  // Return the matrix
  return mat;
}

Eigen::Vector3f h_Eigen_Plane_Solver(Eigen::MatrixXf eigen_cloud, int m, int n)
{
  // Example
  Eigen::MatrixXf matA(m, n);

  // Debugging strange eigen behavior
  matA = eigen_cloud(Eigen::all, Eigen::seqN(0, n));

  Eigen::MatrixXf matB = -1 * Eigen::MatrixXf::Ones(m, 1);

  // Find the plane normal
  Eigen::Vector3f normal = matA.colPivHouseholderQr().solve(matB);

  // Check if the fitting is healthy
  double D = 1 / normal.norm();
  normal.normalize();  // normal is a unit vector from now on
  //bool planeValid = true;
  //for(int i = 0; i < 3; ++i)
  //{  // compare Ax + By + Cz + D with 0.2 (ideally Ax + By + Cz + D = 0)
  //  if(fabs(normal(0) * matA(i, 0) + normal(1) * matA(i, 1) + normal(2) * matA(i, 2) + D) > 0.2)
  //  {
  //    planeValid = false;  // 0.2 is an experimental threshold; can be tuned
  //    break;
  //  }
  //}

  //if(planeValid)
  //{
  //  printf("Eigen solver plane test passed. %d points\n", m);
  //}
  //else
  //{
  //  printf("Eigen solver plane test failed. %d points\n", m);
  //}

  return normal;
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
static int const SAMPLE_WIDTH  = 1280;
static int const SAMPLE_HEIGHT = 720;

// A function that takes a PCL pointcloud of type pcl::PointXYZRGB and returns a PCL pointcloud of type pcl::PointXYZ
pcl::PointCloud<pcl::PointXYZ>::Ptr remove_color(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input)
{
  // Get the number of points
  int n = input->size();

  // Create a new pointcloud of type pcl::PointXYZ
  pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>());
  output->width    = input->width;
  output->height   = input->height;
  output->is_dense = input->is_dense;
  output->points.resize(n);

  // Loop over the input pointcloud and copy the x, y, and z values
  for(int i = 0; i < n; i++)
  {
    output->points[i].x = input->points[i].x;
    output->points[i].y = input->points[i].y;
    output->points[i].z = input->points[i].z;
  }

  // Return the new pointcloud
  return output;
}

// Implicit cube around point cloud
void max_and_min_from_pointcloud(pcl::PointCloud<pcl::PointXYZRGB> cloud, nvmath::vec3f& max, nvmath::vec3f& min) {
  // function takes a PCL pointcloudXYZ as an argument and returns min and max vectors by reference

  // Initialize min and max vectors with extreme values
  min.x = min.y = min.z = std::numeric_limits<float>::max();
  max.x = max.y = max.z = std::numeric_limits<float>::min();

  // Loop over the pointcloud and update min and max vectors
  for(const auto& point : cloud)
  {
    if(point.x < min.x)
      min.x = point.x;
    if(point.y < min.y)
      min.y = point.y;
    if(point.z < min.z)
      min.z = point.z;
    if(point.x > max.x)
      max.x = point.x;
    if(point.y > max.y)
      max.y = point.y;
    if(point.z > max.z)
      max.z = point.z;
  }
}

struct SampleGridDim
{
  // Number of samples in each dimension
  int xdim;
  int ydim;
  int zdim;
};

// Split point cloud into axis-aligned voxel grid
struct SampleGrid
{
  // axis-aligned unit vectors
  vec3 xdir = {1, 0, 0};
  vec3 ydir = {0, 1, 0};
  vec3 zdir = {0, 0, 1};

  // voxel memory address stride in each direction
  unsigned int xstride;
  unsigned int ystride;
  unsigned int zstride;

  // voxel size in x,y,z world coordinates
  float step_size = 0.05;

  // Number of voxels in x,y,z
  SampleGridDim dim;

  // Min and max points in point cloud
  vec3 min;
  vec3 max;

  // Array of normals, with memory layout [z][y][x], where x is fastest moving voxel index
  vec3 *normals;
};

SampleGrid create_sample_grid(pcl::PointCloud<pcl::PointXYZRGB>& cloud, vec3 min, vec3 max, float step_size) {
  vec3 grid_dimensions = max - min;
  SampleGridDim voxel_dimensions = {(int)ceil(grid_dimensions.x / step_size), 
                                     (int)ceil(grid_dimensions.y / step_size), 
                                     (int)ceil(grid_dimensions.z / step_size)};

  SampleGrid retval;

  retval.dim = voxel_dimensions;
  retval.min = min;
  retval.max = max;
  retval.step_size = step_size;

  // Memory layout of voxels/samples is indexed as [z][y][x] in c
  retval.xstride   = 1;
  retval.ystride   = voxel_dimensions.xdim;
  retval.zstride   = voxel_dimensions.xdim * voxel_dimensions.ydim;

  retval.normals = (vec3*)malloc(voxel_dimensions.xdim * voxel_dimensions.ydim * voxel_dimensions.zdim * sizeof(vec3));

  return retval;
}


//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{
  UNUSED(argc);

  // Test PCL read
  pcl::PointCloud<pcl::PointXYZRGB> cloud = read_PLY();
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr = cloud.makeShared();

  Eigen::MatrixXf position_cloud = cloud.getMatrixXfMap(3, 8, 0).transpose(); 

  //// Test Eigen Plane solver
  //Eigen::Vector3f normal = h_Eigen_Plane_Solver(position_cloud, position_cloud.rows(), position_cloud.cols());

  nvmath::vec3f min;
  nvmath::vec3f max;

  QueryPerformanceFrequency(&Frequency);
  QueryPerformanceCounter(&StartingTime);
  printf("Starting KD map...\n");

  max_and_min_from_pointcloud(cloud, max, min);

  pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
  kdtree.setInputCloud(cloud_ptr);

  QueryPerformanceCounter(&EndingTime);
  ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
  ElapsedMicroseconds.QuadPart *= 1000000;
  ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
  printf("Map completed in %lld s", ElapsedMicroseconds.QuadPart / 1000000);

  // Indices for k nearest neighbors, and distances
  pcl::Indices indices;
  std::vector<float> distances;

  // Find k nearest neighbors
  int k = 16;

  // Choose point for testing
  pcl::PointXYZRGB test_point(min.x, min.y, min.z);

  // Call the nearestKSearch method with the point of interest, k, and the two vectors
  kdtree.nearestKSearch(test_point, k, indices, distances);

  // Setup GLFW window
  glfwSetErrorCallback(onErrorCallback);
  if(!glfwInit())
  {
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME, nullptr, nullptr);

  // Setup camera
  CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
  CameraManip.setLookat({8.440, 9.041, -8.973}, {-2.462, 3.661, -0.286}, {0.000, 1.000, 0.000});

  // Setup Vulkan
  if(!glfwVulkanSupported())
  {
    printf("GLFW: Vulkan Not Supported\n");
    return 1;
  }

  // setup some basic things for the sample, logging file for example
  NVPSystem system(PROJECT_NAME);

  // Search path for shaders and other media
  defaultSearchPaths = {
      NVPSystem::exePath() + PROJECT_RELDIRECTORY,
      NVPSystem::exePath() + PROJECT_RELDIRECTORY "..",
      std::string(PROJECT_NAME),
  };

  // Vulkan required extensions
  assert(glfwVulkanSupported() == 1);
  uint32_t count{0};
  auto     reqExtensions = glfwGetRequiredInstanceExtensions(&count);

  // Requesting Vulkan extensions and layers
  nvvk::ContextCreateInfo contextInfo;
  contextInfo.setVersion(1, 2);                       // Using Vulkan 1.2
  for(uint32_t ext_id = 0; ext_id < count; ext_id++)  // Adding required extensions (surface, win32, linux, ..)
    contextInfo.addInstanceExtension(reqExtensions[ext_id]);
  contextInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);              // FPS in titlebar
  contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);  // Allow debug names
  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);            // Enabling ability to present rendering

  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature);  // To use vkCmdTraceRaysKHR
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline

  // Creating Vulkan base application
  nvvk::Context vkctx{};
  vkctx.initInstance(contextInfo);
  // Find all compatible devices
  auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);
  assert(!compatibleDevices.empty());
  // Use a compatible device
  vkctx.initDevice(compatibleDevices[0], contextInfo);

  // Create example
  HelloVulkan helloVk;

  // Window need to be opened to get the surface on which to draw
  const VkSurfaceKHR surface = helloVk.getVkSurface(vkctx.m_instance, window);
  vkctx.setGCTQueueWithPresent(surface);

  helloVk.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
  helloVk.createSwapchain(surface, SAMPLE_WIDTH, SAMPLE_HEIGHT);
  helloVk.createDepthBuffer();
  helloVk.createRenderPass();
  helloVk.createFrameBuffers();

  // Setup Imgui
  helloVk.initGUI(0);  // Using sub-pass 0

  // Creation of the example
  //helloVk.loadModel(nvh::findFile("media/scenes/Medieval_building.obj", defaultSearchPaths, true));
  helloVk.loadModel(nvh::findFile("media/scenes/plane.obj", defaultSearchPaths, true));
  helloVk.loadModel(nvh::findFile("media/scenes/wuson.obj", defaultSearchPaths, true),
                    nvmath::scale_mat4(nvmath::vec3f(0.5f)) * nvmath::translation_mat4(nvmath::vec3f(0.0f, 0.0f, 6.0f)));

  std::random_device              rd;         // Will be used to obtain a seed for the random number engine
  std::mt19937                    gen(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::normal_distribution<float> dis(2.0f, 2.0f);
  std::normal_distribution<float> disn(0.5f, 0.2f);
  auto                            wusonIndex = static_cast<int>(helloVk.m_objModel.size() - 1);
  auto                            planeIndex = 0;

  for(int n = 0; n < 50; ++n)
  {
    ObjInstance inst;
    inst.objIndex       = wusonIndex;
    float         scale = fabsf(disn(gen));
    nvmath::mat4f mat   = nvmath::translation_mat4(nvmath::vec3f{dis(gen), 0.f, dis(gen) + 6});
    //    mat              = mat * nvmath::rotation_mat4_x(dis(gen));
    mat            = mat * nvmath::scale_mat4(nvmath::vec3f(scale));
    inst.transform = mat;

    //helloVk.m_instances.push_back(inst);
  }

  // Creation of implicit geometry
  MaterialObj mat;
  // Reflective
  mat.diffuse   = nvmath::vec3f(0, 0, 0);
  mat.specular  = nvmath::vec3f(1.f);
  mat.shininess = 0.0;
  mat.illum     = 3;
  helloVk.addImplMaterial(mat);
  // Transparent
  mat.diffuse  = nvmath::vec3f(0.4, 0.4, 1);
  mat.illum    = 4;
  mat.dissolve = 0.5;
  helloVk.addImplMaterial(mat);
  //helloVk.addImplCube({-6.1, 0, -6}, {-6, 10, 6}, 0);
  //helloVk.addImplCube(min, max, 0);
  //helloVk.addImplSphere({1, 2, 4}, 1.f, 1);
  

  //// Quick and dirty drawing random sampling from point cloud
  //for(int idx = 0; idx < cloud.size(); idx++)
  //{
  //  bool skip_point = (((double)std::rand() / RAND_MAX) < 0.01f) ? true : false;
  //  if(skip_point) continue;

  //  pcl::PointXYZRGB& point = cloud.at(idx);
  //  helloVk.addImplSphere(nvmath::vec3f(point.x, point.y, point.z), 0.01f, 1);
  //} 

  QueryPerformanceFrequency(&Frequency);
  QueryPerformanceCounter(&StartingTime);
  /*
  * Surface reconstruction, by tangent plane approximation (ref.Hoppe et al.)
  */ 
  // Create sample voxel grid within bounds of point cloud
  float      step_size = 0.1f;
  SampleGrid sGrid     = create_sample_grid(cloud, min, max, step_size);

  printf("\n\nStarting surface reconstruction...\n");
  printf("Grid Dimensions: (x, y, z) = (%d, %d, %d)\n", sGrid.dim.xdim, sGrid.dim.ydim, sGrid.dim.zdim);

  // Get K nearest neighbors, and distances
  pcl::IndicesPtr    indices_ptr(&indices);  // indices of point cloud
  // Compute normal vector for sample by Moving Least Squares of nearest neighbors
  vec3               sample_normal;
  vec3 sample_position;

  // Find k nearest neighbors
  float radius = sGrid.step_size;
  
  // PCL datatypes for filtering data
  pcl::ExtractIndices<pcl::PointXYZRGB>  extract;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGB>);

  // Eigen matrices for doing LLS
  Eigen::MatrixXf eigen_matrix;
  Eigen::Vector3f eigen_normal;

  // Traverse sample grid, and calculate normals at each sample point
  int sample_count = 0;
  for(int zidx = 0; zidx < sGrid.dim.zdim; zidx++)
  {
    for(int yidx = 0; yidx < sGrid.dim.ydim; yidx++)
    {
      for(int xidx = 0; xidx < sGrid.dim.xdim; xidx++)
      {
        // Compute world coordinates of sample point
        sample_position = sGrid.min +
            xidx * sGrid.step_size * sGrid.xdir + yidx * sGrid.step_size * sGrid.ydir + zidx * sGrid.step_size * sGrid.zdir;

        // Sanity check
        if(sample_position.x < sGrid.min.x || sample_position.x > (sGrid.max.x + (sGrid.step_size / 2)) ||
           sample_position.y < sGrid.min.y || sample_position.y > (sGrid.max.y + (sGrid.step_size / 2)) || 
           sample_position.z < sGrid.min.z || sample_position.z > (sGrid.max.z + (sGrid.step_size / 2)))
        {
          printf("Illegal sample position (x, y, z): (%.2f, %.2f, %.2f)\n", sample_position.x, sample_position.y,
                 sample_position.z);
          std::exit(2);
        }

        // Create PCL point object for kdtree search
        pcl::PointXYZRGB pcl_sample_location(sample_position.x, sample_position.y, sample_position.z);

        if(kdtree.radiusSearch(pcl_sample_location, radius, indices, distances) > 3)
        {
          if((indices).size() > 16)
          {
            //printf("%d points in voxel, truncating to nearest %d.\n", (int)(indices).size(), k);

            kdtree.nearestKSearch(pcl_sample_location, k, indices, distances);
          }

          // Extract points from cloud PCL filters
          extract.setInputCloud(cloud_ptr);
          extract.setIndices(indices_ptr);
          extract.setNegative(false);
          extract.filter(*output);

          //printf("Matrix size into MLS is %d\n", (int)output->size());

          eigen_matrix = (*output).getMatrixXfMap(3, 8, 0).transpose();

          eigen_normal = h_Eigen_Plane_Solver(eigen_matrix, (int)(output)->size(), 3);

          sample_normal = {eigen_normal[0], eigen_normal[1], eigen_normal[2]};

          /* 
           * Add planes to world space
           */
          ObjInstance inst;
          inst.objIndex       = planeIndex;
          float         scale = 0.05 / 20.0f; // scale plane to the size of one voxel
          nvmath::mat4f mat   = nvmath::translation_mat4(nvmath::vec3f{sample_position.x, sample_position.y, sample_position.z});

          nvmath::vec3f default_plane_normal = {0, 1, 0};
          float yaw_radians = nvmath::dot(vec3(sample_normal.x, 0, sample_normal.z), default_plane_normal);  // rotate around y axis
          float pitch_radians = nvmath::dot(vec3(0, sample_normal.y, sample_normal.z), default_plane_normal); // rotate around x axis
          float roll_radians = nvmath::dot(vec3(sample_normal.x, sample_normal.y, 0), default_plane_normal);  // rotate around z axis

          yaw_radians = acosf(yaw_radians);
          pitch_radians = acosf(pitch_radians);
          roll_radians = acosf(roll_radians);

          mat                 = mat * nvmath::rotation_yaw_pitch_roll(yaw_radians, pitch_radians, roll_radians);
          mat                 = mat * nvmath::scale_mat4(nvmath::vec3f(scale));
          inst.transform      = mat;

          helloVk.m_instances.push_back(inst);
          sample_count++;
        }
        else // not enough points in range
        {
          sample_normal = vec3(0, 0, 0);
        }

        // Sanity check
        if((xidx) < 0 || (xidx) > sGrid.dim.xdim || 
            (yidx) < 0 || (yidx) > sGrid.dim.ydim || 
            (zidx) < 0 || (zidx) > sGrid.dim.zdim)
        {
          printf("Illegal sample index (x, y, z): (%d, %d, %d)\n", xidx, yidx, zidx);
          printf("Sample position (x, y, z): (%.2f, %.2f, %.2f)\n", sample_position.x, sample_position.y, sample_position.z);
          std::exit(2);
        }

        sGrid.normals[zidx * sGrid.zstride + yidx * sGrid.ystride + xidx * sGrid.xstride] = sample_normal;
      }
    }
  }
  QueryPerformanceCounter(&EndingTime);
  ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
  ElapsedMicroseconds.QuadPart *= 1000000;
  ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
  printf("Surface reconstruction completed in %lld s\n", ElapsedMicroseconds.QuadPart / 1000000);
  printf("Total plane primitives: %d\n", sample_count);

  helloVk.initOffscreen();
  Offscreen& offscreen = helloVk.offscreen();

  helloVk.createImplictBuffers();


  helloVk.createDescriptorSetLayout();
  helloVk.createGraphicsPipeline();
  helloVk.createUniformBuffer();
  helloVk.createObjDescriptionBuffer();
  helloVk.updateDescriptorSet();

  // #VKRay
  helloVk.initRayTracing();


  nvmath::vec4f clearColor   = nvmath::vec4f(1, 1, 1, 1.00f);
  bool          useRaytracer = true;


  helloVk.setupGlfwCallbacks(window);
  ImGui_ImplGlfw_InitForVulkan(window, true);

  // Main loop
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    if(helloVk.isMinimized())
      continue;

    // Start the Dear ImGui frame
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();


    // Show UI window.
    if(helloVk.showGui())
    {
      ImGuiH::Panel::Begin();
      bool changed = false;
      // Edit 3 floats representing a color
      changed |= ImGui::ColorEdit3("Clear color", reinterpret_cast<float*>(&clearColor));
      // Switch between raster and ray tracing
      changed |= ImGui::Checkbox("Ray Tracer mode", &useRaytracer);
      if(changed)
        helloVk.resetFrame();

      renderUI(helloVk);
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGuiH::Control::Info("", "", "(F10) Toggle Pane", ImGuiH::Control::Flags::Disabled);
      ImGuiH::Panel::End();
    }

    // Start rendering the scene
    helloVk.prepareFrame();

    // Start command buffer of this frame
    auto                   curFrame = helloVk.getCurFrame();
    const VkCommandBuffer& cmdBuf   = helloVk.getCommandBuffers()[curFrame];

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    // Updating camera buffer
    helloVk.updateUniformBuffer(cmdBuf);

    // Clearing screen
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color        = {{clearColor[0], clearColor[1], clearColor[2], clearColor[3]}};
    clearValues[1].depthStencil = {1.0f, 0};

    // Offscreen render pass
    {
      VkRenderPassBeginInfo offscreenRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      offscreenRenderPassBeginInfo.clearValueCount = 2;
      offscreenRenderPassBeginInfo.pClearValues    = clearValues.data();
      offscreenRenderPassBeginInfo.renderPass      = offscreen.renderPass();
      offscreenRenderPassBeginInfo.framebuffer     = offscreen.frameBuffer();
      offscreenRenderPassBeginInfo.renderArea      = {{0, 0}, helloVk.getSize()};

      // Rendering Scene
      if(useRaytracer)
      {
        helloVk.raytrace(cmdBuf, clearColor);
      }
      else
      {
        vkCmdBeginRenderPass(cmdBuf, &offscreenRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        helloVk.rasterize(cmdBuf);
        vkCmdEndRenderPass(cmdBuf);
      }
    }

    // 2nd rendering pass: tone mapper, UI
    {
      VkRenderPassBeginInfo postRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      postRenderPassBeginInfo.clearValueCount = 2;
      postRenderPassBeginInfo.pClearValues    = clearValues.data();
      postRenderPassBeginInfo.renderPass      = helloVk.getRenderPass();
      postRenderPassBeginInfo.framebuffer     = helloVk.getFramebuffers()[curFrame];
      postRenderPassBeginInfo.renderArea      = {{0, 0}, helloVk.getSize()};

      // Rendering tonemapper
      vkCmdBeginRenderPass(cmdBuf, &postRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
      offscreen.draw(cmdBuf, helloVk.getSize());

      // Rendering UI
      ImGui::Render();
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);
      vkCmdEndRenderPass(cmdBuf);
    }

    // Submit for display
    vkEndCommandBuffer(cmdBuf);
    helloVk.submitFrame();
  }

  // Cleanup
  vkDeviceWaitIdle(helloVk.getDevice());

  helloVk.destroyResources();
  helloVk.destroy();
  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
