#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

int main (){
  network<sequential> net;
  
  net.load("lenet-model");
  
  std::string data_dir_path = "MNIST";
  std::vector<label_t> test_labels;
  std::vector<vec_t> test_images;
  
  parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
  parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

  for(int i; i<10; i++){
    auto results = net.predict(test_images[i]);
    std::cout<<"Prediction:"<< results<<std::endl;
    std::cout<<"Label:"<<test_labels[i]<<std::endl;
  }
}