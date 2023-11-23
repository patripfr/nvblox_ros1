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

#ifndef NVBLOX_ROS__NVBLOX_HUMAN_NODE_HPP_
#define NVBLOX_ROS__NVBLOX_HUMAN_NODE_HPP_

#include <deque>
#include <memory>
#include <tuple>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>

#include <nvblox/mapper/multi_mapper.h>
#include <nvblox/semantics/image_projector.h>
#include <nvblox/sensors/pointcloud.h>

#include "nvblox_ros/nvblox_node.hpp"

namespace nvblox {

struct mapperPublisherBundle {
  ros::Publisher pointcloud_publisher;
  ros::Publisher esdf_pointcloud_publisher;
  ros::Publisher voxels_publisher;
  ros::Publisher occupancy_publisher;
  ros::Publisher map_slice_publisher; 
  std::string key;
};

class NvbloxHumanNode : public NvbloxNode {
 public:
  explicit NvbloxHumanNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private);
  virtual ~NvbloxHumanNode() = default;

  // Setup. These are called by the constructor.
  void getParameters();
  void initializeMultiMapper();
  void subscribeToTopics();
  void setupTimers();
  void advertiseTopics();

  // Callbacks for Sensor + Mask
  void depthPlusMaskImageCallback(
      const sensor_msgs::ImageConstPtr& depth_img_ptr,
      const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg,
      const sensor_msgs::ImageConstPtr& mask_img_ptr,
      const sensor_msgs::CameraInfo::ConstPtr& mask_camera_info_msg);
  void colorPlusMaskImageCallback(
      const sensor_msgs::ImageConstPtr& color_img_ptr,
      const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg,
      const sensor_msgs::ImageConstPtr& mask_img_ptr,
      const sensor_msgs::CameraInfo::ConstPtr& mask_camera_info_msg);

  // This is our internal type for passing around images, their matching
  // segmentation masks, as well as the camera intrinsics.
  using ImageSegmentationMaskMsgTuple =
      std::tuple<sensor_msgs::ImageConstPtr, sensor_msgs::CameraInfo::ConstPtr,
                 sensor_msgs::ImageConstPtr, sensor_msgs::CameraInfo::ConstPtr>;

  // Override the depth processing from the base node
  void processDepthQueue(const ros::TimerEvent& /*event*/) override;
  void processColorQueue(const ros::TimerEvent& /*event*/) override;

  // The methods for processing images from the internal queue.
  virtual bool processDepthImage(
      const ImageSegmentationMaskMsgTuple& depth_mask_msg);
  virtual bool processColorImage(
      const ImageSegmentationMaskMsgTuple& color_mask_msg);

  // Publish human data on fixed frequency
  void processHumanEsdf(const ros::TimerEvent& /*event*/);

  // Decay the human occupancy grid on fixed frequency
  void decayHumanOccupancy(const ros::TimerEvent& /*event*/);

 protected:
  // Publish human data (if any subscribers) that helps
  // visualization and debugging.
  void publishHumanDebugOutput();

  // Mapper
  // Holds the map layers and their associated integrators
  // - TsdfLayer, ColorLayer, OccupancyLayer, EsdfLayer, MeshLayer
  std::shared_ptr<MultiMapper> multi_mapper_;

  // Holds the masked mappers for the respective keys
  std::shared_ptr<std::map<std::string, std::shared_ptr<Mapper>>> 
    masked_mappers_;


  // Synchronize: Depth + CamInfo + SegmentationMake + CamInfo
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Image,
      sensor_msgs::CameraInfo>
      mask_time_policy_t;
  std::shared_ptr<message_filters::Synchronizer<mask_time_policy_t>>
      timesync_depth_mask_;
  std::shared_ptr<message_filters::Synchronizer<mask_time_policy_t>>
      timesync_color_mask_;

  // Segmentation mask sub.
  message_filters::Subscriber<sensor_msgs::Image> segmentation_mask_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo>
      segmentation_camera_info_sub_;

  ros::NodeHandle nh_;


  // Publishers
  ros::Publisher human_pointcloud_publisher_;
  ros::Publisher human_esdf_pointcloud_publisher_;
  ros::Publisher combined_esdf_pointcloud_publisher_;
  ros::Publisher human_voxels_publisher_;
  ros::Publisher human_occupancy_publisher_;
  ros::Publisher human_map_slice_publisher_;
  ros::Publisher combined_map_slice_publisher_;
  ros::Publisher depth_frame_overlay_publisher_;
  ros::Publisher color_frame_overlay_publisher_;

  // Holds the publishers for the respective keys
  std::map<std::string, mapperPublisherBundle> masked_publishers_;

  // Timers
  ros::Timer human_occupancy_decay_timer_;
  ros::Timer human_esdf_processing_timer_;

  // Keys of the respective object classes
  std::vector<std::string> keys_;

  // Rates.
  float human_occupancy_decay_rate_hz_ = 10.0f;
  float human_esdf_update_rate_hz_ = 10.0f;

  // Image queues.
  // Note these differ from the base class image queues because they also
  // include segmentation images. The base class queue are disused in the
  // NvbloxHumanNode.
  std::deque<ImageSegmentationMaskMsgTuple> depth_mask_image_queue_;
  std::deque<ImageSegmentationMaskMsgTuple> color_mask_image_queue_;

  // Cache for GPU image
  MonoImage mask_image_;

  // Image queue mutexes.
  std::mutex depth_mask_queue_mutex_;
  std::mutex color_mask_queue_mutex_;

  // Object for back projecting image to a pointcloud.
  DepthImageBackProjector image_back_projector_;

  // Device caches
  Pointcloud human_pointcloud_C_device_;
  Pointcloud human_pointcloud_L_device_;
  Pointcloud human_voxel_centers_L_device_;

  // Caching data of last depth frame for debug outputs
  Camera depth_camera_;
  Transform T_L_C_depth_;
};

}  // namespace nvblox

#endif  // NVBLOX_ROS__NVBLOX_HUMAN_NODE_HPP_
