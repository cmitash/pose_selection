#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <camera_constants.h>
#include <simulation_io.hpp>
#include <chrono>

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


#include <fstream>
#include <vector>
#include <set>
#include <iterator>
#include <string>
#include <algorithm>

#include <mpi.h>

/*
 * A class to create and write profiling data in a csv file.
 */
class CSVWriter
{
	std::string fileName;
	std::string delimeter;
	int linesCount;

public:
	CSVWriter(std::string filename, std::string delm = ",") :
			fileName(filename), delimeter(delm), linesCount(1)
	{}
	/*
	 * Member function to store a range as comma seperated value
	 */
	template<typename T>
	void addDatainRow(T first, T last);
};

/*
 * This Function accepts a range and appends all the elements in the range
 * to the last row, seperated by delimeter (Default is comma)
 */
template<typename T>
void CSVWriter::addDatainRow(T first, T last)
{
	std::fstream file;
	// Open the file in truncate mode if first line else in Append Mode
	file.open(fileName, std::ios::out | (linesCount ? std::ios::app : std::ios::trunc));

	// Iterate over the range and add each lement to file seperated by delimeter.
	for (; first != last; )
	{
		file << *first;
		if (++first != last)
			file << delimeter;
	}
	file << "\n";
	linesCount++;
  // GEORGE via rmate server which maps different files
	// Close the file
	file.close();
}  //  end class for csv-data  profiling

//  Extracting a filename from a path
std::string getFileName(const std::string& s) {

   char sep = '/';

   size_t i = s.rfind(sep, s.length());
   if (i != std::string::npos) {
      return(s.substr(i+1, s.length() - i));
   }

   return("");
}


// Global 
SimExample::Ptr simexample;

// Constants
float depth_scale = 1000.0;
float depth_max = 2.0; // 2m

float explanation_threshold = 0.01; // 1cm
float surface_normal_threshold = 30.0; // degrees

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

  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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

  //cout << "scene dir name is:"  << getFileName(scene_path)  << endl;

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
  cv::Mat K = (cv::Mat_<double>(3, 3) << kCameraFX, 0, kCameraCX, 0, kCameraFY, kCameraCY, 0, 0, 1);
  cv::rgbd::RgbdNormals normals_computer(height, width, CV_32F, K, 5, cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD);

  // Read scene depth image and compute normal
  read_depth_image(scene_depth_image, depth_image_filepath);
  normals_computer(scene_depth_image, scene_normals);
  scene_normals.convertTo(scene_normals3f, CV_32FC3);


  if (world_rank==0) {
  // Creating an object of CSVWriter
	CSVWriter writer("/home/pracsys/repos/experiment-data/pose_selection-mpi.csv");  // profiling 
  }
  auto count_ttc_start = chrono::steady_clock::now();  // profiling

  // Read object poses
  object_pose_file.open (object_pose_filepath, std::ifstream::in);

  Eigen::Matrix4f obj_pose, best_obj_pose;
  int hypothesis_count = 0;
  int best_hypothesis_index = -1;
  float best_score = 0;
  best_obj_pose.setIdentity();


// read the input file and save it in a 2d vector
  std::vector< std::vector<double> > vec;
  double x;
  while  (object_pose_file>>x) {
    std::vector<double> row; // Create an empty row
    for (int j = 0; j < 11; j++) {
      row.push_back(x); // Add an element (column) to the row
      object_pose_file>> x;
    }
    vec.push_back(row); // Add the row to the main vector
}
 object_pose_file.close();

 

std::vector< std::vector<double> >::const_iterator row;
std::vector<double>::const_iterator col; 

hypothesis_count = -1;



for (row = vec.begin(); row != vec.end(); ++row)
{
hypothesis_count +=1;
  if ((hypothesis_count % world_size) == world_rank){
    int i,j = 0; 
   for (col = row->begin(); col != row->end(); ++col)
    {
      if (j>3) {j = 0; i = i+1;}
      obj_pose(i,j) = *col;
      j +=1;
    }
      cv::Mat rendered_depth_image = cv::Mat::zeros(height, width, CV_16UC1); 
      render_scene(scene_, object_mesh, obj_pose, rendered_depth_image); 
      normals_computer(rendered_depth_image, rendered_normals);
      rendered_normals.convertTo(rendered_normals3f, CV_32FC3);
      float score = compute_cost(rendered_depth_image, rendered_normals3f, scene_depth_image, scene_normals3f);
      write_depth_image(rendered_depth_image, scene_path + "/rendered_images/" + std::to_string(hypothesis_count) + ".png");
     
  }
} 

  



/*
    while (object_pose_file >> obj_pose(0,0) >> obj_pose(0,1) >> obj_pose(0,2) >> obj_pose(0,3)
            >> obj_pose(1,0) >> obj_pose(1,1) >> obj_pose(1,2) >> obj_pose(1,3)
            >> obj_pose(2,0) >> obj_pose(2,1) >> obj_pose(2,2) >> obj_pose(2,3)) {

      if ((hypothesis_count % world_size) == world_rank){
      
          // render the depth image
          cv::Mat rendered_depth_image = cv::Mat::zeros(height, width, CV_16UC1);

          //auto count_render_scene_start = chrono::steady_clock::now();  // profiling
          render_scene(scene_, object_mesh, obj_pose, rendered_depth_image);
          //auto count_render_scene_end = chrono::steady_clock::now();  // profiling
          
          // compute surface normal for rendered depth image
          normals_computer(rendered_depth_image, rendered_normals);
          rendered_normals.convertTo(rendered_normals3f, CV_32FC3);
          
          // compute score    
          float score = compute_cost(rendered_depth_image, rendered_normals3f, scene_depth_image, scene_normals3f);

        // commenting out to speedup the experiments
        // std::cout << "hypothesis: " << hypothesis_count << ", score: " << score << std::endl;

        write_depth_image(rendered_depth_image, scene_path + "/rendered_images/" + std::to_string(hypothesis_count) + ".png");
      }
      hypothesis_count++;

      // update score
      // GEORGE - FIX THIS
      
      if(score > best_score){
        best_score = score;
        best_obj_pose = obj_pose;
        best_hypothesis_index = hypothesis_count;
      }
      

    }
*/


 // object_pose_file.close();

  if (world_rank==0){

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


    auto count_ttc_end = chrono::steady_clock::now();  // profiling


    // save data
    float ttc = chrono::duration_cast<chrono::milliseconds>(count_ttc_end - count_ttc_start).count(); // time-to-completion
    cout << " total time to completion in microseconds: "  << ttc << endl;

    std::vector<std::string> ttcList = { "TTC", std::to_string(ttc),getFileName(scene_path),std::to_string(object_mesh.polygons.size())};


    // Adding Set to CSV File
    writer.addDatainRow(ttcList.begin(), ttcList.end());

    std::vector<std::string> hcList = {"NumberHypothesis", std::to_string(hypothesis_count),getFileName(scene_path),std::to_string(object_mesh.polygons.size())};
    writer.addDatainRow(hcList.begin(), hcList.end());

  }
  // Finalize the MPI environment.
  MPI_Finalize();


  return 0;
}