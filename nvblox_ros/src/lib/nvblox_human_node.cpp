// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>

#include <nvblox/io/csv.h>

#include "nvblox_ros/nvblox_human_node.hpp"

namespace nvblox {

NvbloxHumanNode::NvbloxHumanNode(ros::NodeHandle& nh,
                                 ros::NodeHandle& nh_private)
    : NvbloxNode(nh, nh_private),
      human_pointcloud_C_device_(MemoryType::kDevice),
      human_pointcloud_L_device_(MemoryType::kDevice) {
  ROS_INFO_STREAM("NvbloxHumanNode::NvbloxHumanNode()");

  // Get parameters specific to the human node.
  getParameters();

  // Initialize the MultiMapper and overwrite the base-class node's Mapper.
  initializeMultiMapper();

  // Add additional timers and publish more topics
  setupTimers();
  advertiseTopics();

  // Subscribe to topics
  // NOTE(alexmillane): This function modifies to base class subscriptions to
  // add synchronization with segmentation masks.
  subscribeToTopics();
}

void NvbloxHumanNode::getParameters() {
  // TODO(TT) how to handle this node?
  nh_private_.getParam("human_occupancy_decay_rate_hz", 
    human_occupancy_decay_rate_hz_);
  nh_private_.getParam("human_esdf_update_rate_hz", human_esdf_update_rate_hz_);
  nh_private_.getParam("keys", keys_);
}

void NvbloxHumanNode::initializeMultiMapper() {
  // Initialize the multi mapper. Composed of:
  // - masked occupancy mapper for humans
  // - unmasked mapper for static objects (with configurable projective layer
  //   type)
  constexpr ProjectiveLayerType kDynamicLayerType =
      ProjectiveLayerType::kOccupancy;
  multi_mapper_ = std::make_shared<MultiMapper>(keys_,
      voxel_size_, MemoryType::kDevice, kDynamicLayerType,
      static_projective_layer_type_);

  // Over-write the base-class node's mapper with the unmasked mapper of
  // the multi mapper. We also have to initialize it from ROS 2 params by
  // calling initializeMapper() (again) (it its also called in the base
  // constructor, on the now-deleted Mapper).
  mapper_ = multi_mapper_.get()->unmasked_mapper();
  initializeMapper(mapper_.get(), nh_private_);
  // Set to an invalid depth to ignore human pixels in the unmasked mapper
  // during integration.
  multi_mapper_->setDepthUnmaskedImageInvalidPixel(-1.f);

  // Initialize the human mapper (masked mapper of the multi mapper)

  masked_mappers_ = multi_mapper_.get()->masked_mappers();
  // Human mapper params have not been declared yet
  // declareMapperParameters(mapper_name, this);
  float max_integration_distance_m;
  for (const auto pair : *masked_mappers_) {
    initializeMapper(pair.second.get(), nh_private_);
    max_integration_distance_m =
      pair.second->occupancy_integrator().max_integration_distance_m();
    masked_publishers_.insert(std::make_pair(pair.first, 
                                             mapperPublisherBundle()));
  }

  // Set to a distance bigger than the max. integration distance to not include
  // non human pixels on the human mapper, but clear along the projection.
  // TODO(remosteiner): Think of a better way to do this.
  // Currently this leads to blocks being allocated even behind solid obstacles.
  multi_mapper_->setDepthMaskedImageInvalidPixel(-1.f);
  // multi_mapper_->setDepthMaskedImageInvalidPixel(
  //     max_integration_distance_m * 2.f);
}

void NvbloxHumanNode::subscribeToTopics() {
  ROS_INFO_STREAM("NvbloxHumanNode::subscribeToTopics()");

  // Increased queue size compared to the NvbloxNode,
  // because of bigger delay comming from segmentation.
  constexpr int kQueueSize = 40;

  // Unsubscribe from base-class synchronized topics.
  // We redo synchronization below.
  NvbloxNode::timesync_depth_.reset();
  NvbloxNode::timesync_color_.reset();

  segmentation_mask_sub_.subscribe(nh_, "mask/image", 20);
  segmentation_camera_info_sub_.subscribe(nh_, "mask/camera_info", 20);

  if (use_depth_) {
    // Unsubscribe from the depth topic in nvblox_node
    timesync_depth_.reset();
    // Subscribe to depth + mask + cam_infos
    timesync_depth_mask_ =
        std::make_shared<message_filters::Synchronizer<mask_time_policy_t>>(
            mask_time_policy_t(kQueueSize), depth_sub_, depth_camera_info_sub_,
            segmentation_mask_sub_, segmentation_camera_info_sub_);
    timesync_depth_mask_->registerCallback(
        std::bind(&NvbloxHumanNode::depthPlusMaskImageCallback, this,
                  std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4));
  }

  if (use_color_) {
    // Unsubscribe from the color topic in nvblox_node
    timesync_color_.reset();
    // Subscribe to color + mask + cam_infos
    timesync_color_mask_ =
        std::make_shared<message_filters::Synchronizer<mask_time_policy_t>>(
            mask_time_policy_t(kQueueSize), color_sub_, color_camera_info_sub_,
            segmentation_mask_sub_, segmentation_camera_info_sub_);
    timesync_color_mask_->registerCallback(
        std::bind(&NvbloxHumanNode::colorPlusMaskImageCallback, this,
                  std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4));
  }
}

void NvbloxHumanNode::advertiseTopics() {
  // Add some stuff
  for (auto& pair : masked_publishers_) {
    const std::string key = pair.first;
    const std::string pcl_topic = key + "_pointcloud";
    pair.second.pointcloud_publisher =  
      nh_private_.advertise<sensor_msgs::PointCloud2>(pcl_topic, 1, false);
    const std::string esdf_topic = key + "_esdf_pointcloud";
    pair.second.esdf_pointcloud_publisher = 
      nh_private_.advertise<sensor_msgs::PointCloud2>(esdf_topic, 1, false);
    const std::string voxels_topic = key + "_voxels";
    pair.second.voxels_publisher = 
      nh_private_.advertise<visualization_msgs::Marker>(voxels_topic, 1, false);
    const std::string occupancy_topic = key + "_occupancy";
    pair.second.occupancy_publisher = 
      nh_private_.advertise<sensor_msgs::PointCloud2>(
        occupancy_topic, 1, false);
    const std::string map_slice_topic = key + "_map_slice";
    pair.second.map_slice_publisher = 
      nh_private_.advertise<nvblox_msgs::DistanceMapSlice>(map_slice_topic, 1, 
                                                           false);
  }

  combined_esdf_pointcloud_publisher_ =
      nh_private_.advertise<sensor_msgs::PointCloud2>(
          "combined_esdf_pointcloud", 1, false);
  combined_map_slice_publisher_ =
      nh_private_.advertise<nvblox_msgs::DistanceMapSlice>("combined_map_slice",
                                                           1, false);
  depth_frame_overlay_publisher_ = nh_private_.advertise<sensor_msgs::Image>(
      "depth_frame_overlay", 1, false);
  color_frame_overlay_publisher_ = nh_private_.advertise<sensor_msgs::Image>(
      "color_frame_overlay", 1, false);
}

void NvbloxHumanNode::setupTimers() {
  {
    if (human_occupancy_decay_rate_hz_ > 0) {
      ros::TimerOptions timer_options(
          ros::Duration(1.0 / human_occupancy_decay_rate_hz_),
          boost::bind(&NvbloxHumanNode::decayHumanOccupancy, this, _1),
          &processing_queue_);
      human_occupancy_decay_timer_ = nh_private_.createTimer(timer_options);
    }
  }
  {
    ros::TimerOptions timer_options(
        ros::Duration(1.0 / human_esdf_update_rate_hz_),
        boost::bind(&NvbloxHumanNode::processHumanEsdf, this, _1),
        &processing_queue_);
    human_esdf_processing_timer_ = nh_private_.createTimer(timer_options);
  }
}

void NvbloxHumanNode::depthPlusMaskImageCallback(
    const sensor_msgs::ImageConstPtr& depth_img_ptr,
    const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg,
    const sensor_msgs::ImageConstPtr& mask_img_ptr,
    const sensor_msgs::CameraInfo::ConstPtr& mask_camera_info_msg) {
  pushMessageOntoQueue<ImageSegmentationMaskMsgTuple>(
      std::make_tuple(depth_img_ptr, camera_info_msg, mask_img_ptr,
                      mask_camera_info_msg),
      &depth_mask_image_queue_, &depth_mask_queue_mutex_);
}

void NvbloxHumanNode::colorPlusMaskImageCallback(
    const sensor_msgs::ImageConstPtr& color_img_ptr,
    const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg,
    const sensor_msgs::ImageConstPtr& mask_img_ptr,
    const sensor_msgs::CameraInfo::ConstPtr& mask_camera_info_msg) {
  pushMessageOntoQueue<ImageSegmentationMaskMsgTuple>(
      std::make_tuple(color_img_ptr, camera_info_msg, mask_img_ptr,
                      mask_camera_info_msg),
      &color_mask_image_queue_, &color_mask_queue_mutex_);
}

void NvbloxHumanNode::processDepthQueue(const ros::TimerEvent& /*event*/) {
  auto message_ready = [this](const ImageSegmentationMaskMsgTuple& msg) {
    return this->canTransform(std::get<0>(msg)->header) &&
           this->canTransform(std::get<2>(msg)->header);
  };
  processMessageQueue<ImageSegmentationMaskMsgTuple>(
      &depth_mask_image_queue_,  // NOLINT
      &depth_mask_queue_mutex_,  // NOLINT
      message_ready,             // NOLINT
      std::bind(&NvbloxHumanNode::processDepthImage, this,
                std::placeholders::_1));

  limitQueueSizeByDeletingOldestMessages(maximum_sensor_message_queue_length_,
                                         "depth_mask", &depth_mask_image_queue_,
                                         &depth_mask_queue_mutex_);
}

void NvbloxHumanNode::processColorQueue(const ros::TimerEvent& /*event*/) {
  auto message_ready = [this](const ImageSegmentationMaskMsgTuple& msg) {
    return this->canTransform(std::get<0>(msg)->header) &&
           this->canTransform(std::get<2>(msg)->header);
  };
  processMessageQueue<ImageSegmentationMaskMsgTuple>(
      &color_mask_image_queue_,  // NOLINT
      &color_mask_queue_mutex_,  // NOLINT
      message_ready,             // NOLINT
      std::bind(&NvbloxHumanNode::processColorImage, this,
                std::placeholders::_1));

  limitQueueSizeByDeletingOldestMessages(maximum_sensor_message_queue_length_,
                                         "color_mask", &color_mask_image_queue_,
                                         &color_mask_queue_mutex_);
}

bool NvbloxHumanNode::processDepthImage(
    const ImageSegmentationMaskMsgTuple& depth_mask_msg) {
  timing::Timer ros_total_timer("ros/total");
  timing::Timer ros_depth_timer("ros/depth");
  timing::Timer transform_timer("ros/depth/transform");

  // Message parts
  const sensor_msgs::ImageConstPtr& depth_img_ptr = std::get<0>(depth_mask_msg);
  const sensor_msgs::CameraInfo::ConstPtr& depth_camera_info_msg =
      std::get<1>(depth_mask_msg);
  const sensor_msgs::ImageConstPtr& mask_img_ptr = std::get<2>(depth_mask_msg);
  const sensor_msgs::CameraInfo::ConstPtr& mask_camera_info_msg =
      std::get<3>(depth_mask_msg);

  // Check that we're not updating more quickly than we should.
  if (isUpdateTooFrequent(depth_img_ptr->header.stamp, last_depth_update_time_,
                          max_depth_update_hz_)) {
    return true;
  }
  last_depth_update_time_ = depth_img_ptr->header.stamp;

  // Get the TF for BOTH images.
  const std::string depth_img_frame = depth_img_ptr->header.frame_id;
  if (!transformer_.lookupTransformToGlobalFrame(
          depth_img_frame, depth_img_ptr->header.stamp, &T_L_C_depth_)) {
    return false;
  }
  Transform T_L_C_mask;
  const std::string mask_img_frame = mask_img_ptr->header.frame_id;
  if (!transformer_.lookupTransformToGlobalFrame(
          mask_img_frame, mask_img_ptr->header.stamp, &T_L_C_mask)) {
    return false;
  }
  Transform T_CM_CD = T_L_C_mask.inverse() * T_L_C_depth_;
  if (is_realsense_data_) {
    // There is an unresolved issue with the ROS realsense wrapper.
    // Until it is fixed, the below inverse needs to be applied.
    // https://github.com/IntelRealSense/realsense-ros/issues/2500
    T_CM_CD = T_CM_CD.inverse();
  }
  transform_timer.Stop();

  timing::Timer conversions_timer("ros/depth/conversions");
  // Convert camera info message to camera object.
  depth_camera_ = conversions::cameraFromMessage(*depth_camera_info_msg);
  const Camera mask_camera =
      conversions::cameraFromMessage(*mask_camera_info_msg);

  // Convert the depth image.
  if (!conversions::depthImageFromImageMessage(depth_img_ptr, &depth_image_)) {
    ROS_ERROR("Failed to transform depth or mask image.");
    return false;
  }



  if (mask_img_ptr->encoding != "mono8") {
    return false;
  }
  cv_bridge::CvImageConstPtr mono_cv_image =
      cv_bridge::toCvCopy(mask_img_ptr, "mono8");


  std::vector<cv::Mat> channels;
  channels.resize(keys_.size()+1);
  for (size_t i=0; i<channels.size(); i++) {
    channels[i] = cv::Mat::zeros(mono_cv_image->image.rows,
                  mono_cv_image->image.cols,
                  CV_8UC1);
  }

  // Uncomment this line to separate masked objects from the background
  // channels[0] = mono_cv_image->image.clone();
  
  for (int u=0; u<mono_cv_image->image.rows; u++) {
    for (int v=0; v<mono_cv_image->image.cols; v++) {
      uint8_t val = mono_cv_image->image.at<uint8_t>(u,v);
      if (val < channels.size() && val > 0u) {
        channels[int(val)].at<uint8_t>(u,v) = val;
      }
    }
  }
  conversions_timer.Stop();

  timing::Timer integration_timer("ros/depth/integrate");


  for (size_t i=0; i<channels.size(); i++) {
    timing::Timer integration_timer("ros/depth/integrate/conversion");
    if (!conversions::monoImageFromCVImage(channels[i], &mask_image_)) {
      ROS_ERROR("Failed to transform color or mask image.");
      return false;
    }
    integration_timer.Stop();

    // Integrate
    if (i == 0) {
      multi_mapper_->setMinDepthImage(depth_image_, mask_image_,
                                    T_CM_CD, depth_camera_, mask_camera);
      timing::Timer depth_integration_timer("ros/depth/integrate/depth");
      multi_mapper_->integrateDepthFromMin(depth_image_, mask_image_, T_L_C_depth_,
                                T_CM_CD, depth_camera_, mask_camera);
      depth_integration_timer.Stop();
    } else {
      timing::Timer mask_integration_timer("ros/depth/integrate/masked");
      multi_mapper_->integrateDepthMaskedFromMin(depth_image_, mask_image_, 
                                T_L_C_depth_,T_CM_CD, depth_camera_, 
                                mask_camera, keys_[i-1]);
      mask_integration_timer.Stop();                        
    }
  }

  integration_timer.Stop();
  return true;
}

bool NvbloxHumanNode::processColorImage(
    const ImageSegmentationMaskMsgTuple& color_mask_msg) {
  timing::Timer ros_total_timer("ros/total");
  timing::Timer ros_color_timer("ros/color");
  timing::Timer transform_timer("ros/color/transform");

  // Message parts
  const sensor_msgs::ImageConstPtr& color_img_ptr = std::get<0>(color_mask_msg);
  const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg =
      std::get<1>(color_mask_msg);
  const sensor_msgs::ImageConstPtr& mask_img_ptr = std::get<2>(color_mask_msg);
  const sensor_msgs::CameraInfo::ConstPtr& mask_camera_info_msg =
      std::get<3>(color_mask_msg);

  // Check that we're not updating more quickly than we should.
  if (isUpdateTooFrequent(color_img_ptr->header.stamp, last_color_update_time_,
                          max_color_update_hz_)) {
    return true;
  }
  last_color_update_time_ = color_img_ptr->header.stamp;

  // Get the TF for BOTH images.
  Transform T_L_C;
  const std::string color_img_frame = color_img_ptr->header.frame_id;
  if (!transformer_.lookupTransformToGlobalFrame(
          color_img_frame, color_img_ptr->header.stamp, &T_L_C)) {
    return false;
  }
  Transform T_L_C_mask;
  const std::string mask_img_frame = mask_img_ptr->header.frame_id;
  if (!transformer_.lookupTransformToGlobalFrame(
          mask_img_frame, mask_img_ptr->header.stamp, &T_L_C_mask)) {
    return false;
  }
  transform_timer.Stop();

  timing::Timer conversions_timer("ros/color/conversions");
  // Convert camera info message to camera object.
  const Camera color_camera = conversions::cameraFromMessage(*camera_info_msg);
  const Camera mask_camera =
      conversions::cameraFromMessage(*mask_camera_info_msg);
  if (!camerasAreEquivalent(color_camera, mask_camera, T_L_C, T_L_C_mask)) {
    ROS_ERROR(
        "Color and mask image are not coming from the same camera or frame.");
    return false;
  }

  // // Convert the color image.
  if (!conversions::colorImageFromImageMessage(color_img_ptr, &color_image_)) {
      ROS_ERROR("Failed to transform color or mask image.");
      return false;
    }

  if (mask_img_ptr->encoding != "mono8") {
    return false;
  }
  cv_bridge::CvImageConstPtr mono_cv_image =
      cv_bridge::toCvCopy(mask_img_ptr, "mono8");

  std::vector<cv::Mat> channels;
  channels.resize(keys_.size()+1);
  for (size_t i=0; i<channels.size(); i++) {
    channels[i] = cv::Mat::zeros(mono_cv_image->image.rows,
                  mono_cv_image->image.cols,
                  CV_8UC1);
  }

  // Uncomment this line to separate masked objects from the background
  // channels[0] = mono_cv_image->image.clone();

  for (int u=0; u<mono_cv_image->image.rows; u++) {
    for (int v=0; v<mono_cv_image->image.cols; v++) {
      uint8_t val = mono_cv_image->image.at<uint8_t>(u,v);
      if (val < channels.size() && val > 0) {
        channels[val].at<uint8_t>(u,v) = 255;
      }
    }
  }
  conversions_timer.Stop();

  timing::Timer integration_timer("ros/color/integrate");
  for (size_t i=0; i<channels.size(); i++) {
    if (!conversions::monoImageFromCVImage(channels[i], &mask_image_)) {
      ROS_ERROR("Failed to transform color or mask image.");
      return false;
    }

    // Integrate
    if (i == 0) {
      multi_mapper_->integrateColor(color_image_, mask_image_, T_L_C, 
                                    color_camera);
    } else {
      multi_mapper_->integrateColorMasked(color_image_, mask_image_, T_L_C, 
                                          color_camera, keys_[i-1]);
    }

  }

  integration_timer.Stop();
  return true;
}

void NvbloxHumanNode::processHumanEsdf(const ros::TimerEvent& /*event*/) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  timing::Timer ros_total_timer("ros/total");
  timing::Timer ros_human_total_timer("ros/humans");

  if (last_depth_update_time_.toSec() <= 0.f) {
    return;  // no data yet.
  }
  publishHumanDebugOutput();

  // DISABLED, NEEDS TO BE ADAPTED FOR MULITPLE CLASSES
  // // Process the human esdf layer.
  // timing::Timer esdf_integration_timer("ros/humans/esdf/integrate");
  // std::vector<Index3D> updated_blocks;
  // if (esdf_2d_) {
  //   updated_blocks = masked_mappers_->updateEsdfSlice(
  //       esdf_2d_min_height_, esdf_2d_max_height_, esdf_slice_height_);
  // } else {
  //   updated_blocks = masked_mappers_->updateEsdf();
  // }
  // esdf_integration_timer.Stop();

  // if (updated_blocks.empty()) {
  //   return;
  // }

  // timing::Timer esdf_output_timer("ros/humans/esdf/output");

  // // Check if anyone wants any human slice
  // if (esdf_distance_slice_ &&
  //         (human_esdf_pointcloud_publisher_.getNumSubscribers() > 0) ||
  //     (human_map_slice_publisher_.getNumSubscribers() > 0)) {
  //   // Get the slice as an image
  //   timing::Timer esdf_slice_compute_timer("ros/humans/esdf/output/compute");
  //   AxisAlignedBoundingBox aabb;
  //   Image<float> map_slice_image;
  //   esdf_slice_converter_.distanceMapSliceImageFromLayer(
  //       masked_mappers_->esdf_layer(), esdf_slice_height_, &map_slice_image,
  //       &aabb);
  //   esdf_slice_compute_timer.Stop();

  //   // Human slice pointcloud (for visualization)
  //   if (human_esdf_pointcloud_publisher_.getNumSubscribers() > 0) {
  //     timing::Timer esdf_output_human_pointcloud_timer(
  //         "ros/humans/esdf/output/pointcloud");
  //     sensor_msgs::PointCloud2 pointcloud_msg;
  //     esdf_slice_converter_.sliceImageToPointcloud(
  //         map_slice_image, aabb, esdf_slice_height_,
  //         masked_mappers_->esdf_layer().voxel_size(), &pointcloud_msg);
  //     pointcloud_msg.header.frame_id = global_frame_;
  //     pointcloud_msg.header.stamp = ros::Time::now();
  //     human_esdf_pointcloud_publisher_.publish(pointcloud_msg);
  //   }

  //   // Human slice (for navigation)
  //   if (human_map_slice_publisher_.getNumSubscribers() > 0) {
  //     timing::Timer esdf_output_human_slice_timer(
  //         "ros/humans/esdf/output/slice");
  //     nvblox_msgs::DistanceMapSlice map_slice_msg;
  //     esdf_slice_converter_.distanceMapSliceImageToMsg(
  //         map_slice_image, aabb, esdf_slice_height_,
  //         masked_mappers_->voxel_size_m(), &map_slice_msg);
  //     map_slice_msg.header.frame_id = global_frame_;
  //     map_slice_msg.header.stamp = ros::Time::now();
  //     human_map_slice_publisher_.publish(map_slice_msg);
  //   }
  // }

  // // Check if anyone wants any human+statics slice
  // if (esdf_distance_slice_ &&
  //         (combined_esdf_pointcloud_publisher_.getNumSubscribers() > 0) ||
  //     (combined_map_slice_publisher_.getNumSubscribers() > 0)) {
  //   // Combined slice
  //   timing::Timer esdf_slice_compute_timer(
  //       "ros/humans/esdf/output/combined/compute");
  //   Image<float> combined_slice_image;
  //   AxisAlignedBoundingBox combined_aabb;
  //   esdf_slice_converter_.distanceMapSliceFromLayers(
  //       mapper_->esdf_layer(), masked_mappers_->esdf_layer(), 
  //       esdf_slice_height_, &combined_slice_image, &combined_aabb);
  //   esdf_slice_compute_timer.Stop();

  //   // Human+Static slice pointcloud (for visualization)
  //   if (combined_esdf_pointcloud_publisher_.getNumSubscribers() > 0) {
  //     timing::Timer esdf_output_human_pointcloud_timer(
  //         "ros/humans/esdf/output/combined/pointcloud");
  //     sensor_msgs::PointCloud2 pointcloud_msg;
  //     esdf_slice_converter_.sliceImageToPointcloud(
  //         combined_slice_image, combined_aabb, esdf_slice_height_,
  //         masked_mappers_->esdf_layer().voxel_size(), &pointcloud_msg);
  //     pointcloud_msg.header.frame_id = global_frame_;
  //     pointcloud_msg.header.stamp = ros::Time::now();
  //     combined_esdf_pointcloud_publisher_.publish(pointcloud_msg);
  //   }

  //   // Human+Static slice (for navigation)
  //   if (combined_map_slice_publisher_.getNumSubscribers() > 0) {
  //     timing::Timer esdf_output_human_slice_timer(
  //         "ros/humans/esdf/output/combined/slice");
  //     nvblox_msgs::DistanceMapSlice map_slice_msg;
  //     esdf_slice_converter_.distanceMapSliceImageToMsg(
  //         combined_slice_image, combined_aabb, esdf_slice_height_,
  //         masked_mappers_->voxel_size_m(), &map_slice_msg);
  //     map_slice_msg.header.frame_id = global_frame_;
  //     map_slice_msg.header.stamp = ros::Time::now();
  //     human_map_slice_publisher_.publish(map_slice_msg);
  //   }
  // }
  // esdf_output_timer.Stop();
}

void NvbloxHumanNode::decayHumanOccupancy(
    const ros::TimerEvent& /*event*/) {
  std::unique_lock<std::mutex> lock(map_mutex_);
  for (auto pair : *masked_mappers_) {
    pair.second->decayOccupancy();
  }
}

void NvbloxHumanNode::publishHumanDebugOutput() {
  timing::Timer ros_human_debug_timer("ros/humans/output/debug");

  // DISABLED FOR THE MOMENT, NEEDS TO BE ADAPTED TO WORK WITH MULTIPLE CLASSES
  // Get a human pointcloud
  // if (human_pointcloud_publisher_.getNumSubscribers() +
  //         human_voxels_publisher_.getNumSubscribers() >
  //     0) {
  //   // Grab the human only image.
  //   const DepthImage& depth_image_only_humans =
  //       multi_mapper_->getLastDepthFrameMasked();
  //   // Back project
  //   image_back_projector_.backProjectOnGPU(
  //      depth_image_only_humans, depth_camera_, &human_pointcloud_C_device_,
  //      masked_mappers_->occupancy_integrator().max_integration_distance_m());
  //   transformPointcloudOnGPU(T_L_C_depth_, human_pointcloud_C_device_,
  //                            &human_pointcloud_L_device_);
  // }

  // // Publish the human pointcloud
  // if (human_pointcloud_publisher_.getNumSubscribers() > 0) {
  //   // Back-project human depth image to pointcloud and publish.
  //   sensor_msgs::PointCloud2 pointcloud_msg;
  //   pointcloud_converter_.pointcloudMsgFromPointcloud(
  //       human_pointcloud_L_device_, &pointcloud_msg);
  //   pointcloud_msg.header.frame_id = global_frame_;
  //   pointcloud_msg.header.stamp = ros::Time::now();
  //   human_pointcloud_publisher_.publish(pointcloud_msg);
  // }

  // Publish human voxels
  // if (human_voxels_publisher_.getNumSubscribers() > 0) {
  //   // Human voxels from points (in the layer frame)
  //   image_back_projector_.pointcloudToVoxelCentersOnGPU(
  //       human_pointcloud_L_device_, voxel_size_,
  //       &human_voxel_centers_L_device_);
  //   // Publish
  //   visualization_msgs::Marker marker_msg;
  //   pointcloud_converter_.pointsToCubesMarkerMsg(
  //       human_voxel_centers_L_device_.points().toVector(), voxel_size_,
  //       Color::Red(), &marker_msg);
  //   marker_msg.header.frame_id = global_frame_;
  //   marker_msg.header.stamp = ros::Time::now();
  //   human_voxels_publisher_.publish(marker_msg);
  // }

  for (auto& key : *masked_mappers_)
    // Publish the class occupancy layer
    if (masked_publishers_[key.first].occupancy_publisher.getNumSubscribers() > 
        0) {
      sensor_msgs::PointCloud2 pointcloud_msg;
      layer_converter_.pointcloudMsgFromLayer(key.second->occupancy_layer(),
                                              &pointcloud_msg);
      pointcloud_msg.header.frame_id = global_frame_;
      pointcloud_msg.header.stamp = ros::Time::now();
      masked_publishers_[key.first].occupancy_publisher.publish(pointcloud_msg);
    }
}

}  // namespace nvblox