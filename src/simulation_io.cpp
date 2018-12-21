#include <simulation_io.hpp>
#include <camera_constants.h>
#include <pcl/io/png_io.h>

#include <opencv2/core/core.hpp>

pcl::simulation::SimExample::SimExample(int argc, char **argv,
                                        int height, int width):
  height_(height), width_(width) {

  initializeGL (argc, argv);

  // 1. construct member elements:
  camera_ = Camera::Ptr (new Camera ());
  scene_ = Scene::Ptr (new Scene ());

  rl_ = RangeLikelihood::Ptr (new RangeLikelihood (1, 1, height, width, scene_));

  // Actually corresponds to default parameters:
  rl_->setCameraIntrinsicsParameters (width_, height_, kCameraFX,
                                      kCameraFY, kCameraCX, kCameraCY);
  rl_->setComputeOnCPU (false);

  rl_->setSumOnCPU (false);
  rl_->setUseColor (false);

  for (int i = 0; i < 2048; i++) {
    float v = i / 2048.0;
    v = powf(v, 3) * 6;
    t_gamma[i] = v * 6 * 256;
  }
}


void
pcl::simulation::SimExample::initializeGL (int argc, char **argv) {
  glutInit (&argc, argv);
  glutInitDisplayMode (GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);// was GLUT_RGBA
  glutInitWindowPosition (10, 10);
  glutInitWindowSize (10, 10);
  //glutInitWindowSize (window_width_, window_height_);
  glutCreateWindow ("OpenGL range likelihood");
  GLenum err = glewInit ();

  if (GLEW_OK != err) {
    std::cerr << "Error: " << glewGetErrorString (err) << std::endl;
    exit (-1);
  }

  std::cout << "Status: Using GLEW " << glewGetString (GLEW_VERSION) <<
            std::endl;

  if (glewIsSupported ("GL_VERSION_2_0")) {
    std::cout << "OpenGL 2.0 supported" << std::endl;
  } else {
    std::cerr << "Error: OpenGL 2.0 not supported" << std::endl;
    exit(1);
  }

  std::cout << "GL_MAX_VIEWPORTS: " << GL_MAX_VIEWPORTS << std::endl;
  const GLubyte *version = glGetString (GL_VERSION);
  std::cout << "OpenGL Version: " << version << std::endl;
}



void
pcl::simulation::SimExample::doSim (Eigen::Isometry3d pose_in) {
  /*
  // No reference image - but this is kept for compatability with range_test_v2:
  float *reference = new float[rl_->getRowHeight() * rl_->getColWidth()];
  const float *depth_buffer = rl_->getDepthBuffer();

  // Copy one image from our last as a reference.
  for (int i = 0, n = 0; i < rl_->getRowHeight(); ++i) {
    for (int j = 0; j < rl_->getColWidth(); ++j) {
      reference[n++] = depth_buffer[i * rl_->getWidth() + j];
    }
  }
*/

  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>
                                                                           poses;
  //std::vector<float> scores;
  //int n = 1;
  poses.push_back (pose_in);
 // rl_->computeLikelihoods (reference, poses, scores);
 rl_->render(poses);

 // delete [] reference;
}



void
pcl::simulation::SimExample::write_score_image(const float *score_buffer,
                                               std::string fname) {
  int npixels = rl_->getWidth() * rl_->getHeight();
  uint8_t *score_img = new uint8_t[npixels * 3];

  float min_score = score_buffer[0];
  float max_score = score_buffer[0];

  for (int i = 1; i < npixels; i++) {
    if (score_buffer[i] < min_score) {
      min_score = score_buffer[i];
    }

    if (score_buffer[i] > max_score) {
      max_score = score_buffer[i];
    }
  }

  for (int y = 0; y <  height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int i = y * width_ + x ;
      int i_in = (height_ - 1 - y) * width_ + x ; // flip up

      float d = (score_buffer[i_in] - min_score) / (max_score - min_score);
      score_img[3 * i + 0] = 0;
      score_img[3 * i + 1] = d * 255;
      score_img[3 * i + 2] = 0;
    }
  }

  // Write to file:
  pcl::io::saveRgbPNGFile (fname, score_img, width_, height_);

  delete [] score_img;
}

void
pcl::simulation::SimExample::write_depth_image(const float *depth_buffer,
                                               std::string fname) {
  int npixels = rl_->getWidth() * rl_->getHeight();
  uint8_t *depth_img = new uint8_t[npixels * 3];

  // float min_depth = depth_buffer[0];
  // float max_depth = depth_buffer[0];
  //
  // for (int i = 1; i < npixels; i++) {
  //   if (depth_buffer[i] < min_depth) {
  //     min_depth = depth_buffer[i];
  //   }
  //
  //   if (depth_buffer[i] > max_depth) {
  //     max_depth = depth_buffer[i];
  //   }
  // }

  #pragma omp parallel for
  for (int y = 0; y <  height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int i = y * width_ + x ;
      int i_in = (height_ - 1 - y) * width_ + x ; // flip up down


      float zn = 0.1; //ZNEAR
      float zf = 20.0;
      float d = depth_buffer[i_in];
      float z = -zf * zn / ((zf - zn) * (d - zf / (zf - zn)));
      float b = 0.075;
      float f = 580.0;
      uint16_t kd = static_cast<uint16_t>(1090 - b * f / z * 8);

      if (kd < 0) {
        kd = 0;
      } else if (kd > 2047) {
        kd = 2047;
      }

      int pval = t_gamma[kd];
      int lb = pval & 0xff;

      switch (pval >> 8) {
      case 0:
        depth_img[3 * i + 0] = 255;
        depth_img[3 * i + 1] = 255 - lb;
        depth_img[3 * i + 2] = 255 - lb;
        break;

      case 1:
        depth_img[3 * i + 0] = 255;
        depth_img[3 * i + 1] = lb;
        depth_img[3 * i + 2] = 0;
        break;

      case 2:
        depth_img[3 * i + 0] = 255 - lb;
        depth_img[3 * i + 1] = 255;
        depth_img[3 * i + 2] = 0;
        break;

      case 3:
        depth_img[3 * i + 0] = 0;
        depth_img[3 * i + 1] = 255;
        depth_img[3 * i + 2] = lb;
        break;

      case 4:
        depth_img[3 * i + 0] = 0;
        depth_img[3 * i + 1] = 255 - lb;
        depth_img[3 * i + 2] = 255;
        break;

      case 5:
        depth_img[3 * i + 0] = 0;
        depth_img[3 * i + 1] = 0;
        depth_img[3 * i + 2] = 255 - lb;
        break;

      default:
        depth_img[3 * i + 0] = 0;
        depth_img[3 * i + 1] = 0;
        depth_img[3 * i + 2] = 0;
        break;
      }
    }
  }

  // Write to file:
  pcl::io::saveRgbPNGFile (fname, depth_img, width_, height_);

  delete [] depth_img;
}


void
pcl::simulation::SimExample::write_depth_image_uint(const float *depth_buffer,
                                                    std::string fname) {
  int npixels = rl_->getWidth() * rl_->getHeight();
  unsigned short *depth_img = new unsigned short[npixels ];

  // float min_depth = depth_buffer[0];
  // float max_depth = depth_buffer[0];
  //
  // for (int i = 1; i < npixels; i++) {
  //   if (depth_buffer[i] < min_depth) {
  //     min_depth = depth_buffer[i];
  //   }
  //
  //   if (depth_buffer[i] > max_depth) {
  //     max_depth = depth_buffer[i];
  //   }
  // }

  #pragma omp parallel for
  for (int y = 0; y <  height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int i = y * width_ + x ;
      int i_in = (height_ - 1 - y) * width_ + x ; // flip up down

      float zn = 0.1; //ZNEAR
      float zf = 20.0;
      float d = depth_buffer[i_in];

      unsigned short z_new = (unsigned short)  round( 1000 * ( -zf * zn / ((
                                                                             zf - zn) * (d - zf / (zf - zn))))); //ZNEAR

      if (z_new < 0) {
        z_new = 0;
      } else if (z_new > 65535) {
        z_new = 65535;
      }

      if ( z_new < 18000) {
        //cout << z_new << " " << d << " " << x << "\n";
      }

      float z = 1000 * ( -zf * zn / ((zf - zn) * (d - zf / (zf - zn))));
      float b = 0.075;
      float f = 580.0;
      uint16_t kd = static_cast<uint16_t>(1090 - b * f / z * 8);

      if (kd < 0) {
        kd = 0;
      } else if (kd > 2047) {
        kd = 2047;
      }

      int pval = t_gamma[kd];
      int lb = pval & 0xff;
      depth_img[i] = z_new;
    }
  }

  // Write to file:
  pcl::io::saveShortPNGFile (fname, depth_img, width_, height_, 1);

  delete [] depth_img;
}

void
pcl::simulation::SimExample::get_depth_image_uint(const float *depth_buffer,
                                                  std::vector<unsigned short> *depth_img) {
  int npixels = rl_->getWidth() * rl_->getHeight();
  depth_img->clear();
  depth_img->resize(npixels);

  // float min_depth = depth_buffer[0];
  // float max_depth = depth_buffer[0];
  //
  // for (int i = 1; i < npixels; i++) {
  //   if (depth_buffer[i] < min_depth) {
  //     min_depth = depth_buffer[i];
  //   }
  //
  //   if (depth_buffer[i] > max_depth) {
  //     max_depth = depth_buffer[i];
  //   }
  // }

  #pragma omp parallel for
  for (int y = 0; y <  height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int i = y * width_ + x ;
      int i_in = (height_ - 1 - y) * width_ + x ; // flip up down

      float zn = 0.1; //ZNEAR
      float zf = 20.0;
      float d = depth_buffer[i_in];

      unsigned short z_new = (unsigned short)  round( 1000 * ( -zf * zn / ((
                                                                             zf - zn) * (d - zf / (zf - zn))))); //ZNEAR

      if (z_new < 0) {
        z_new = 0;
      } else if (z_new > 65535) {
        z_new = 65535;
      }

      if ( z_new < 18000) {
        //cout << z_new << " " << d << " " << x << "\n";
      }

      float z = 1000 * ( -zf * zn / ((zf - zn) * (d - zf / (zf - zn))));
      float b = 0.075;
      float f = 580.0;
      uint16_t kd = static_cast<uint16_t>(1090 - b * f / z * 8);

      if (kd < 0) {
        kd = 0;
      } else if (kd > 2047) {
        kd = 2047;
      }

      int pval = t_gamma[kd];
      int lb = pval & 0xff;
      (*depth_img)[i] = z_new;
    }
  }
}

void pcl::simulation::SimExample::get_depth_image_cv(const float *depth_buffer,
                                                     cv::Mat &depth_image) {
  int npixels = rl_->getWidth() * rl_->getHeight();
  // depth_image.create(height_, width_, CV_16UC1);

  // float min_depth = depth_buffer[0];
  // float max_depth = depth_buffer[0];
  //
  // for (int i = 1; i < npixels; i++) {
  //   if (depth_buffer[i] < min_depth) {
  //     min_depth = depth_buffer[i];
  //   }
  //
  //   if (depth_buffer[i] > max_depth) {
  //     max_depth = depth_buffer[i];
  //   }
  // }

  #pragma omp parallel for
  for (int y = 0; y <  height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int i = y * width_ + x ;
      int i_in = (height_ - 1 - y) * width_ + x ; // flip up down

      float zn = 0.1; //ZNEAR
      float zf = 20.0;
      float d = depth_buffer[i_in];

      unsigned short z_new = (unsigned short)  round( 1000 * ( -zf * zn / ((
                                                                             zf - zn) * (d - zf / (zf - zn))))); //ZNEAR

      if (z_new < 0) {
        z_new = 0;
      } else if (z_new > 65535) {
        z_new = 65535;
      }

      if ( z_new < 18000) {
        //cout << z_new << " " << d << " " << x << "\n";
      }

      float z = 1000 * ( -zf * zn / ((zf - zn) * (d - zf / (zf - zn))));
      float b = 0.075;
      float f = 580.0;
      uint16_t kd = static_cast<uint16_t>(1090 - b * f / z * 8);

      if (kd < 0) {
        kd = 0;
      } else if (kd > 2047) {
        kd = 2047;
      }

      int pval = t_gamma[kd];
      int lb = pval & 0xff;
      depth_image.at<unsigned short>(y, x) = z_new;
    }
  }
}

void
pcl::simulation::SimExample::write_rgb_image(const uint8_t *rgb_buffer,
                                             std::string fname) {
  int npixels = rl_->getWidth() * rl_->getHeight();
  uint8_t *rgb_img = new uint8_t[npixels * 3];

  for (int y = 0; y <  height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      int px = y * width_ + x ;
      int px_in = (height_ - 1 - y) * width_ + x ; // flip up down
      rgb_img [3 * (px) + 0] = rgb_buffer[3 * px_in + 0];
      rgb_img [3 * (px) + 1] = rgb_buffer[3 * px_in + 1];
      rgb_img [3 * (px) + 2] = rgb_buffer[3 * px_in + 2];
    }
  }

  // Write to file:
  pcl::io::saveRgbPNGFile (fname, rgb_img, width_, height_);

  delete [] rgb_img;
}



