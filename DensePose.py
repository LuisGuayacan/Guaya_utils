import os
from os.path import exists, join, basename, splitext
import subprocess

def cmd(command, path='.'):
  if type(command)==str:
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE,cwd=path)
  else:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, cwd=path)
  output, error = process.communicate()
  

def densepose_install():
  git_repo_url = 'https://github.com/facebookresearch/DensePose.git'
  project_name = splitext(basename(git_repo_url))[0]

  c1 = "wget -q https://repo.anaconda.com/archive/Anaconda2-2019.03-Linux-x86_64.sh"
  c2 = "chmod +x Anaconda2-2019.03-Linux-x86_64.sh"
  c3 = "bash ./Anaconda2-2019.03-Linux-x86_64.sh -b -f -p /content/anaconda2"
  c4 = "conda install -y pyyaml=3.12"
  c5 = "conda install -y mkl-include"
  c6 = "conda install -y pytorch=1.0.1 torchvision cudatoolkit=10.0 -c pytorch"
  c7 = "ln -s /content/anaconda2/lib/python2.7/site-packages/torch/lib/ /content/anaconda2/lib/python2.7/site-packages/"
  c8 = "conda install -y -c serge-sans-paille gcc_49"
  c9 = "ln -fs /content/anaconda2/lib/libmpfr.so /content/anaconda2/lib/libmpfr.so.4"
  c10 = "conda install -y protobuf=3.5"
  c11 = "conda install -y -c conda-forge pycocotools"
  c12 = "pip install opencv-python==4.0.0.21 memory_profiler"
  c13 = "git clone -q --depth 1 --recursive -b v1.0.1 https://github.com/pytorch/pytorch"
  c14 = ['git', 'clone', '-q', '--depth', '1', git_repo_url]
  c15 = "pip install -q -r requirements.txt"
  c16 = "make"
  c17 = "make ops"
  c18 = "bash get_densepose_uv.sh"
  c19 = ["sed", "-i", '413s/./\t\t#&/', "vis.py"]
  c20 = ["sed", "-i", '419s/./\t\t#&/', "vis.py"]

  # install Anaconda Python 2.7 to control the dependencies
  # see for more info: 
  if not exists('anaconda2'):
    print('Installing anaconda2...')
    print('Please, be patient. This may take about 20 minutes', end='\n\n')
    cmd(c1)
    cmd(c2)
    cmd(c3)
    # set PATH environment variable
    os.environ['PATH'] = "/content/anaconda2/bin:" + os.environ['PATH']
    # install PyTorch
    cmd(c4)
    cmd(c5)
    cmd(c6)
    cmd(c7)
    # install GCC 4.9
    cmd(c8)
    cmd(c9)
    os.environ['CC'] = '/content/anaconda2/bin/gcc-4.9'
    os.environ['CXX'] = '/content/anaconda2/bin/g++-4.9'
    # protobuf 3.5
    #!apt-get -qq remove -y protobuf-compiler
    cmd(c10)
    # pycocotools
    cmd(c11)
    # some missing dependencies
    cmd(c12)

  # we need some headers from the pytorch source
  if not exists('pytorch'):
    print('Installing pytorch...', end='\n\n')
    cmd(c13)

  ################### DENSE POSE ##################
  if not exists(project_name):
    print('Downloading DensePose...', end='\n\n')
    # clone project
    cmd(c14)

    # install dependencies
    cmd(c15, project_name)

    # update CMakeLists.txt
    cmakelists_txt_content = """
  cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
  set(Caffe2_DIR "/content/anaconda2/lib/python2.7/site-packages/torch/share/cmake/Caffe2/")
  find_package(Caffe2 REQUIRED)

  include_directories("/content/anaconda2/lib/python2.7/site-packages/torch/lib/include")
  include_directories("/content/anaconda2/include")
  include_directories("/content/pytorch")

  add_library(libprotobuf STATIC IMPORTED)
  set(PROTOBUF_LIB "/content/anaconda2/lib/libprotobuf.a")
  set_property(TARGET libprotobuf PROPERTY IMPORTED_LOCATION "${PROTOBUF_LIB}")

  if (${CAFFE2_VERSION} VERSION_LESS 0.8.2)
    # Pre-0.8.2 caffe2 does not have proper interface libraries set up, so we
    # will rely on the old path.
    message(WARNING
        "You are using an older version of Caffe2 (version " ${CAFFE2_VERSION}
        "). Please consider moving to a newer version.")
    include(cmake/legacy/legacymake.cmake)
    return()
  endif()

  # Add compiler flags.
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -fPIC -Wno-narrowing")

  # Print configuration summary.
  include(cmake/Summary.cmake)
  detectron_print_config_summary()

  # Collect custom ops sources.
  file(GLOB CUSTOM_OPS_CPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/detectron/ops/*.cc)
  file(GLOB CUSTOM_OPS_GPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/detectron/ops/*.cu)

  # Install custom CPU ops lib.
  add_library(
      caffe2_detectron_custom_ops SHARED
      ${CUSTOM_OPS_CPU_SRCS})

  target_link_libraries(caffe2_detectron_custom_ops caffe2_library libprotobuf)
  install(TARGETS caffe2_detectron_custom_ops DESTINATION lib)

  # Install custom GPU ops lib, if gpu is present.
  if (CAFFE2_USE_CUDA OR CAFFE2_FOUND_CUDA)
    # Additional -I prefix is required for CMake versions before commit (< 3.7):
    # https://github.com/Kitware/CMake/commit/7ded655f7ba82ea72a82d0555449f2df5ef38594
    list(APPEND CUDA_INCLUDE_DIRS -I${CAFFE2_INCLUDE_DIRS})
    CUDA_ADD_LIBRARY(
        caffe2_detectron_custom_ops_gpu SHARED
        ${CUSTOM_OPS_CPU_SRCS}
        ${CUSTOM_OPS_GPU_SRCS})

    target_link_libraries(caffe2_detectron_custom_ops_gpu caffe2_gpu_library libprotobuf)
    install(TARGETS caffe2_detectron_custom_ops_gpu DESTINATION lib)
  endif()"""
    open(join(project_name, 'CMakeLists.txt'), 'w').write(cmakelists_txt_content)
    # build
    cmd(c16, project_name)
    cmd(c17, project_name)
    # download dense pose data
    cmd(c18, 'DensePose/DensePoseData')
  cmd(c19, 'DensePose/detectron/utils')
  cmd(c20, 'DensePose/detectron/utils')
  print('DensePose is ready to be used!')


# Compute DensePose for a single image
def densepose_img(img_inpath, img_outpath='/content/output'):
  dpExecute = ['python2', 'tools/infer_simple.py', 
               '--cfg', 'configs/DensePose_ResNet101_FPN_s1x-e2e.yaml', 
               '--output-dir', img_outpath, 
               '--image-ext', 'jpg', 
               '--wts', 'https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl',
               img_inpath]
  if not os.path.isdir(img_outpath):
    os.makedirs(img_outpath)
  cmd(dpExecute, 'DensePose')
  

# Compute DensePose for a single video
def densepose_vid(vid_inpath, vid_outpath='/content/output'):
  import cv2
  import numpy as np
  video_name = vid_inpath.split("/")[-1]
  frames_path = os.path.join(vid_outpath, video_name.split(".")[0])
  if not os.path.isdir(frames_path):
    os.makedirs(frames_path)
  
  densepose_path = os.path.join(vid_outpath, video_name.split(".")[0]+"_DP")
  if not os.path.isdir(densepose_path):
      os.makedirs(densepose_path)

  cam = cv2.VideoCapture(vid_inpath)
  currentframe = 0

  while(True): 
    ret,frame = cam.read() 

    if ret: 
      nz = 4-len(str(currentframe))
      name = os.path.join(".",frames_path,"frame"+"0"*nz+str(currentframe)+".jpg")
      cv2.imwrite(name, frame) 
      currentframe += 1
    else:
      break
  
  cam.release() 
  cv2.destroyAllWindows()
  
  for imagefile in np.sort(os.listdir(frames_path)):
    img_inpath = os.path.join(frames_path, imagefile)
    img_outpath = densepose_path
    densepose_img(img_inpath, img_outpath)


# Compute DensePose for each element in a directory
def densepose_dir(dir_inpath, dir_outpath='/content/output'):
  import numpy as np
  img_ext_list = ['jpg','png','bmp','tif','jpeg']
  vid_ext_list = ['mp4','mov','avi','wmv','flv']
  for file in np.sort(os.listdir(dir_inpath)):
    ext = file.split('.')[-1]
    origin_path = os.path.join(dir_inpath,file)
    if ext.lower() in img_ext_list: 
      densepose_img(origin_path, dir_outpath)
    elif ext.lower() in vid_ext_list: 
      densepose_vid(origin_path, dir_outpath)
    else:
      print('The file '+file+' is not an image or video')


def densepose_compute(origin_path, output_path='/content/output'):
  img_ext_list = ['jpg','png','bmp','tif','jpeg']
  vid_ext_list = ['mp4','mov','avi','wmv','flv']
  ext = origin_path.split(".")[-1]

  if ext.lower() in vid_ext_list:
    densepose_vid(origin_path, output_path)
  elif ext.lower() in img_ext_list:
    densepose_img(origin_path, output_path)
  elif os.path.isdir(origin_path):
    densepose_dir(origin_path, output_path)
  else:
    print("The origin_path is invalid")