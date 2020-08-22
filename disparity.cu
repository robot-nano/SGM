//
// Created by wserver on 2020/7/23.
//

#include "disparity.h"
#include <opencv2/highgui/highgui.hpp>

static cudaStream_t stream1, stream2, stream3;
static uint8_t *d_im0;
static uint8_t *d_im1;
static cost_t *d_transform0;
static cost_t *d_transform1;
static uint8_t *d_cost;
static uint8_t *d_disparity;
static uint8_t *d_disparity_filtered_uchar;
static uint8_t *h_disparity;
static uint16_t *d_S;
static uint8_t *d_L0;
static uint8_t *d_L1;
static uint8_t *d_L2;
static uint8_t *d_L3;
static uint8_t *d_L4;
static uint8_t *d_L5;
static uint8_t *d_L6;
static uint8_t *d_L7;
static uint8_t p1_, p2_;
static bool first_alloc;
static uint32_t cols, rows, size, size_cube_l;
static uint16_t *tmp_trans0, *tmp_trans1;

void init_disparity_method(const uint8_t p1, const uint8_t p2) {
  CUDA_CHECK_RETURN(cudaStreamCreate(&stream1));
  CUDA_CHECK_RETURN(cudaStreamCreate(&stream2));
  CUDA_CHECK_RETURN(cudaStreamCreate(&stream3));
  first_alloc = true;
  p1_ = p1;
  p2_ = p2;
  rows = 0;
  cols = 0;
}

cv::Mat compute_disparity_method(cv::Mat &left, cv::Mat &right) {
  if (cols != left.cols || rows != left.rows) {
    debug_log("WARNING: cols or rows are different");
    if (!first_alloc) {
      debug_log("Freeing memory");
      free_memory();
    }
    first_alloc = false;
    cols = left.cols;
    rows = left.rows;
    size = rows * cols;
    size_cube_l = size * MAX_DISPARITY;

    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_transform0, sizeof(cost_t) * size));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_transform1, sizeof(cost_t) * size));

    int size_cube = size * MAX_DISPARITY;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_cost, sizeof(uint8_t) * size_cube));

    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_im0, sizeof(uint8_t) * size));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_im1, sizeof(uint8_t) * size));

    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_S, sizeof(uint16_t) * size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_L0, sizeof(uint8_t) * size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_L1, sizeof(uint8_t) * size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_L2, sizeof(uint8_t) * size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_L3, sizeof(uint8_t) * size_cube_l));
#if PATH_AGGREGATION == 8
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_L4, sizeof(uint8_t) * size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_L5, sizeof(uint8_t) * size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_L6, sizeof(uint8_t) * size_cube_l));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_L7, sizeof(uint8_t) * size_cube_l));
#endif

    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_disparity, sizeof(uint8_t) * size));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_disparity_filtered_uchar, sizeof(uint8_t) * size));

    h_disparity = new uint8_t[size];
    tmp_trans0 = new uint16_t[size];
    tmp_trans1 = new uint16_t[size];
  }
  debug_log("Copying images to the GPU");
  CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im0, left.ptr<uint8_t>(),
                                    sizeof(uint8_t) * size, cudaMemcpyHostToDevice, stream1));
  CUDA_CHECK_RETURN(cudaMemcpyAsync(d_im1, right.ptr<uint8_t>(),
                                    sizeof(uint8_t) * size, cudaMemcpyHostToDevice, stream1));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  dim3 block_size;
  block_size.x = WARP_SIZE;
  block_size.y = WARP_SIZE;

  dim3 grid_size;
  grid_size.x = (cols + block_size.x - 1) / block_size.x;
  grid_size.y = (rows + block_size.y - 1) / block_size.y;

  debug_log("Calling CSCT");
  CensusKernelSM2<<<grid_size, block_size, 0, stream1>>>(d_im0, d_im1, d_transform0, d_transform1, rows, cols);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s %d\n", cudaGetErrorString(err), err);
    exit(-1);
  }

  // Hamming distance
  CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
  debug_log("Calling Hamming Distance");
  HammingDistanceCostKernel<<<rows, MAX_DISPARITY, 0, stream1>>>(d_transform0, d_transform1, d_cost, rows, cols);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s %d\n", cudaGetErrorString(err), err);
    exit(-1);
  }

  // Cost Aggregation
  const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE / WARP_SIZE;
  const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ / WARP_SIZE;

  debug_log("Calling Left to Right");
  CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream2>>>(
      d_cost, d_L0,
      p1_, p2_, rows, cols,
      d_transform0, d_transform1,
      d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s %d\n", cudaGetErrorString(err), err);
    exit(-1);
  }
  debug_log("Calling Right to Left");
  CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0, stream3>>>(
      d_cost, d_L1,
      p1_, p2_, rows, cols,
      d_transform0, d_transform1,
      d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s %d\n", cudaGetErrorString(err), err);
    exit(-1);
  }
  debug_log("Calling Up to Down");
  CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(
      d_cost, d_L2,
      p1_, p2_, rows, cols,
      d_transform0, d_transform1,
      d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s %d\n", cudaGetErrorString(err), err);
    exit(-1);
  }
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  debug_log("Callingg Down to Up");
  CostAggregationKernelDownToUp<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(
      d_cost, d_L3,
      p1_, p2_, rows, cols,
      d_transform0, d_transform1,
      d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s %d\n", cudaGetErrorString(err), err);
    exit(-1);
  }

#if PATH_AGGREGATION == 8
  CostAggregationKernelDiagonalDownUpLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(
      d_cost, d_L4,
      p1_, p2_, rows, cols,
      d_transform0, d_transform1,
      d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s %d\n", cudaGetErrorString(err), err);
    exit(-1);
  }

  CostAggregationKernelDiagonalUpDownLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(
      d_cost, d_L5,
      p1_, p2_, rows, cols,
      d_transform0, d_transform1,
      d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s %d\n", cudaGetErrorString(err), err);
    exit(-1);
  }

  CostAggregationKernelDiagonalDownUpRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(
      d_cost, d_L6,
      p1_, p2_, rows, cols,
      d_transform0, d_transform1,
      d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s %d\n", cudaGetErrorString(err), err);
    exit(-1);
  }

  CostAggregationKernelDiagonalUpDownRightLeft<<<(cols+PIXELS_PER_BLOCK)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0, stream1>>>(
      d_cost, d_L7,
      p1_, p2_, rows, cols,
      d_transform0, d_transform1,
      d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);
#endif

  uint8_t *disp = new uint8_t[rows * cols];
//  uint8_t *tmp = new uint8_t[rows * cols * MAX_DISPARITY];
  CUDA_CHECK_RETURN(cudaMemcpy(disp, d_disparity, sizeof(uint8_t) * rows * cols, cudaMemcpyDeviceToHost));
//  for (int i = 0; i < rows * cols; ++i) {
//    uint8_t min_val = UINT8_MAX;
//    for (int j = 0; j < MAX_DISPARITY; ++j) {
//      min_val = std::min(min_val, tmp[i * MAX_DISPARITY + j]);
//    }
//    disp[i] = min_val * 50;
//  }
  cv::Mat disp_img(rows, cols, CV_8UC1, disp);
  cv::imshow("img", disp_img);
  cv::imwrite("../img/result.png", disp_img);
  cv::waitKey(0);
  return disp_img;
}

static void free_memory() {
  CUDA_CHECK_RETURN(cudaFree(d_im0));
  CUDA_CHECK_RETURN(cudaFree(d_im1));
  CUDA_CHECK_RETURN(cudaFree(d_transform0));
  CUDA_CHECK_RETURN(cudaFree(d_transform1));

  CUDA_CHECK_RETURN(cudaFree(d_cost));
}