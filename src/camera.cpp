#include <iostream>
#include <camera.h>
#include <camera_constants.h>

using namespace Eigen;
using namespace pcl::simulation;

// For LINEMOD dataset
int kCameraWidth = 640;
int kCameraHeight = 480;
float kCameraFX = 572.4114f;
float kCameraFY = 573.57043f;
float kCameraCX = 325.2611f;
float kCameraCY = 242.04899f;
float kZNear = 0.1f;
float kZFar = 20.0f;

void
pcl::simulation::Camera::move (double vx, double vy, double vz)
{
  Vector3d v;
  v << vx, vy, vz;
  pose_.pretranslate (pose_.rotation ()*v);
  x_ = pose_.translation ().x ();
  y_ = pose_.translation ().y ();
  z_ = pose_.translation ().z ();
}

void
pcl::simulation::Camera::updatePose ()
{
  Matrix3d m;
  m = AngleAxisd (yaw_, Vector3d::UnitZ ())
    * AngleAxisd (pitch_, Vector3d::UnitY ())
    * AngleAxisd (roll_, Vector3d::UnitX ());

  pose_.setIdentity ();
  pose_ *= m;
  
  Vector3d v;
  v << x_, y_, z_;
  pose_.translation () = v;
}

void
pcl::simulation::Camera::setParameters (int width, int height,
                                        float fx, float fy,
                                        float cx, float cy,
                                        float z_near, float z_far)
{
  width_ = width;
  height_ = height;
  fx_ = fx;
  fy_ = fy;
  cx_ = cx;
  cy_ = cy;
  z_near_ = z_near;
  z_far_ = z_far;

  float z_nf = (z_near_-z_far_);
  projection_matrix_ <<  2.0f*fx_/width_,  0,                 1.0f-(2.0f*cx_/width_),     0,
                         0,                2.0f*fy_/height_,  1.0f-(2.0f*cy_/height_),    0,
                         0,                0,                (z_far_+z_near_)/z_nf,  2.0f*z_near_*z_far_/z_nf,
                         0,                0,                -1.0f,                  0;
}

void
pcl::simulation::Camera::initializeCameraParameters ()
{
  setParameters (kCameraWidth, kCameraHeight,
                 kCameraFX, kCameraFY,
                 kCameraCX, kCameraCY,
                 kZNear, kZFar); //ZNEAR

  std::cout << "Camera params:: kCameraWidth " << kCameraWidth << " " << 
            "kCameraHeight: " << kCameraHeight << " " << 
            "kCameraFX: " << kCameraFX << " " << 
            "kCameraFY: " << kCameraFY << " " << 
            "kCameraCX: " << kCameraCX << " " << 
            "kCameraCY: " << kCameraCY << " " << 
            "kZNear: " << kZNear << " " << 
            "kZFar: " << kZFar << " " << std::endl;

}
