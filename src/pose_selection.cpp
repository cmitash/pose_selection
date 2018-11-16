#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <camera_constants.h>
#include <simulation_io.hpp>

// For OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd.hpp>

#include <pcl/io/ply_io.h>

using namespace Eigen;
using namespace pcl;
using namespace pcl::console;
using namespace pcl::io;
using namespace pcl::simulation;
using namespace std;

// Global 
SimExample::Ptr simexample;

// Constants
float depth_scale = 1000.0;
float depth_max = 2.0; // 2m

float explanation_threshold = 0.01; // 1cm
float surface_normal_threshold = 30.0; // degrees

float fx = 572.4114f;
float cx = 325.2611f;
float fy = 573.57043f;
float cy = 242.04899f;

void 
read_depth_image(cv::Mat &depth_image, 
                std::string depth_image_filepath) {

    cv::Mat depth_image_raw = cv::imread(depth_image_filepath, CV_16UC1);
    depth_image = cv::Mat::zeros(depth_image_raw.rows, depth_image_raw.cols, CV_32FC1);

    for(int u = 0; u < depth_image_raw.rows; u++)
      for(int v = 0; v< depth_image_raw.cols; v++) {
        unsigned short depth_short = depth_image_raw.at<unsigned short>(u,v);
        depth_image.at<float>(u, v) = (float)depth_short/depth_scale;
      }
}

void 
write_depth_image(cv::Mat &depth_image, 
                  std::string depth_image_filepath) {

  cv::Mat depth_image_raw = cv::Mat::zeros(depth_image.rows, depth_image.cols, CV_16UC1);
  
  for(int u = 0; u < depth_image.rows; u++)
    for(int v = 0; v < depth_image.cols; v++){
      float depth = depth_image.at<float>(u,v)*depth_scale;
      depth_image_raw.at<unsigned short>(u, v) = (unsigned short)depth;
    }

  cv::imwrite(depth_image_filepath, depth_image_raw);
}

void 
transform_poly_mesh(const pcl::PolygonMesh::Ptr &mesh_in, 
                  pcl::PolygonMesh::Ptr &mesh_out, 
                  Eigen::Matrix4f &transform) {

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::fromPCLPointCloud2(mesh_in->cloud, *cloud_in);
  transformPointCloud(*cloud_in, *cloud_out, transform);
  
  *mesh_out = *mesh_in;
  
  pcl::toPCLPointCloud2(*cloud_out, mesh_out->cloud);
}

float 
compute_cost(cv::Mat& rendered_depth_image,
            cv::Mat_<cv::Vec3f>& rendered_surface_normal, 
            cv::Mat& scene_depth_image,
            cv::Mat_<cv::Vec3f>& scene_surface_normal) {

    float score = 0;

    for (int i = 0; i < scene_depth_image.rows; ++i)
    {
      float* pObs = scene_depth_image.ptr<float>(i);
      float* pRen = rendered_depth_image.ptr<float>(i);
      for (int j = 0; j < scene_depth_image.cols; ++j)
      {
          float obVal = *pObs++;
          float renVal = *pRen++;

          float absDiff = fabs(obVal - renVal);

          // needs to be changed when modifying optimization direction
          if(obVal > 0 && renVal > 0 && absDiff < explanation_threshold) {

            cv::Vec3f vec1 = rendered_surface_normal(i, j), vec2 = scene_surface_normal(i, j);
            vec1 = vec1 / cv::norm(vec1);
            vec2 = vec2 / cv::norm(vec2);

            float dot = vec1.dot(vec2);
            float normal_diff = std::min(std::acos(dot), std::acos(-dot));
            normal_diff = normal_diff*180.0/M_PI;

            if(normal_diff < surface_normal_threshold)
              score += 1;
          }
      }
    }

    return score;      
}

void 
render_scene(pcl::simulation::Scene::Ptr scene_ptr,
            pcl::PolygonMesh& object_mesh,
            Eigen::Matrix4f& obj_pose,
            cv::Mat& depth_image) {

  Eigen::Isometry3d camera_pose;
  camera_pose.setIdentity();

  // TODO::this initial camera transform is to counter the difference in camera axes of the simulator. Can we get rid of this?
  Matrix3d m;
  m = AngleAxisd(0, Vector3d::UnitZ())     * AngleAxisd(0, Vector3d::UnitY())    * AngleAxisd(M_PI/2, Vector3d::UnitX()); 
  camera_pose *= m;
  m = AngleAxisd(M_PI/2, Vector3d::UnitZ())     * AngleAxisd(0, Vector3d::UnitY())    * AngleAxisd(0, Vector3d::UnitX()); 
  camera_pose *= m;

  scene_ptr->clear();

  pcl::PolygonMesh::Ptr mesh_in (new pcl::PolygonMesh (object_mesh));
  pcl::PolygonMesh::Ptr mesh_out (new pcl::PolygonMesh (object_mesh));

  transform_poly_mesh(mesh_in, mesh_out, obj_pose);

  PolygonMeshModel::Ptr transformed_mesh = PolygonMeshModel::Ptr (new PolygonMeshModel (GL_POLYGON, mesh_out));
  scene_ptr->add (transformed_mesh);

  simexample->doSim(camera_pose);

  const float *depth_buffer = simexample->rl_->getDepthBuffer();
  simexample->get_depth_image_cv(depth_buffer, depth_image);

  depth_image.convertTo(depth_image, CV_32FC1);
  depth_image = depth_image/depth_scale;
  depth_image.setTo(0,depth_image > depth_max);
}

int
main (int argc, char** argv)
{

  int width = kCameraWidth;
  int height = kCameraHeight;

  pcl::PolygonMesh object_mesh;
  cv::Mat scene_depth_image;
  cv::Mat scene_normal_image;
  std::ifstream object_pose_file;

  cv::Mat rendered_normals, scene_normals;
  cv::Mat_<cv::Vec3f> rendered_normals3f, scene_normals3f;

  std::string scene_path;

  if(argc < 2) {
    std::cout << "Enter the scene path!!!" << std::endl;
    exit(-1);
  }

  scene_path = std::string(argv[1]);

  std::string object_pose_filepath = scene_path + "/pose_hypotheses.txt";
  std::string depth_image_filepath = scene_path + "/depth.png";
  std::string object_model_path = scene_path + "/model.obj";

  // set up camera simulation
  simexample = SimExample::Ptr (new SimExample (argc, argv, height, width));
  pcl::simulation::Scene::Ptr scene_;
  scene_ = simexample->scene_;
  if (scene_ == NULL) {
    printf("ERROR: Scene is not set\n");
  }

  // Read object model file. This step can be performed once and the model can be kept in memory.
  pcl::io::loadPolygonFile(object_model_path, object_mesh);

  // Setup surface normal computer
  cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  cv::rgbd::RgbdNormals normals_computer(height, width, CV_32F, K, 5, cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD);

  // Read scene depth image and compute normal
  read_depth_image(scene_depth_image, depth_image_filepath);
  normals_computer(scene_depth_image, scene_normals);
  scene_normals.convertTo(scene_normals3f, CV_32FC3);

  // Read object poses
  object_pose_file.open (object_pose_filepath, std::ifstream::in);

  Eigen::Matrix4f obj_pose, best_obj_pose;
  int hypothesis_count = 0;
  int best_hypothesis_index = -1;
  float best_score = 0;
  best_obj_pose.setIdentity();

  while (object_pose_file >> obj_pose(0,0) >> obj_pose(0,1) >> obj_pose(0,2) >> obj_pose(0,3)
            >> obj_pose(1,0) >> obj_pose(1,1) >> obj_pose(1,2) >> obj_pose(1,3)
            >> obj_pose(2,0) >> obj_pose(2,1) >> obj_pose(2,2) >> obj_pose(2,3)) {

    // render the depth image
    cv::Mat rendered_depth_image = cv::Mat::zeros(height, width, CV_16UC1);
    render_scene(scene_, object_mesh, obj_pose, rendered_depth_image);

    // compute surface normal for rendered depth image
    normals_computer(rendered_depth_image, rendered_normals);
    rendered_normals.convertTo(rendered_normals3f, CV_32FC3);
    
    // compute score
    float score = compute_cost(rendered_depth_image, rendered_normals3f, scene_depth_image, scene_normals3f);

    std::cout << "hypothesis: " << hypothesis_count << ", score: " << score << std::endl;

    // update score
    if(score > best_score){
      best_score = score;
      best_obj_pose = obj_pose;
      best_hypothesis_index = hypothesis_count;
    }

    write_depth_image(rendered_depth_image, scene_path + "/rendered_images/" + std::to_string(hypothesis_count) + ".png");
    hypothesis_count++;
  }
  object_pose_file.close();

  Eigen::Matrix3f rotm;
  rotm  << best_obj_pose(0,0) ,best_obj_pose(0,1) ,best_obj_pose(0,2)
        ,best_obj_pose(1,0) ,best_obj_pose(1,1) ,best_obj_pose(1,2) 
        ,best_obj_pose(2,0) ,best_obj_pose(2,1) ,best_obj_pose(2,2);

  Eigen::Quaternionf rotq(rotm);

  std::cout << "Best_pose index: " << best_hypothesis_index
            << ", Best score: " << best_score 
            << std::endl
            << "Best pose:" << std::endl 
            << best_obj_pose(0,3) << " " << best_obj_pose(1,3) << " " << best_obj_pose(2,3) << " " 
            << rotq.w() << " " << rotq.x() << " " << rotq.y() << " " << rotq.z() << std::endl;

  return 0;
}