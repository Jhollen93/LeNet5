#include "tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::layers;

network<sequential> net;

net << conv(32, 32, 5, 1, 6) << sigmoid_layer() 
    << ave_pool(28, 28, 6, 2)
    << conv(14, 14, 5, 6, 16) << sigmoid_layer()
    << ave_pool(10, 10, 16, 2)
    << conv(5, 5, 5, 16, 120) << sigmoid_layer()
    << fc(120, 84) << sigmoid_layer()
    << fc(84, 10) << softmax();

std::string data_dir_path = "path_to_your_data_directory";
std::vector<label_t> train_labels, test_labels;
std::vector<vec_t> train_images, test_images;

parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

adagrad optimizer;
net.train<cross_entropy>(optimizer, train_images, train_labels, batch_size, num_epochs);

size_t batch_size = 10; // Example batch size
int num_epochs = 30; // Example number of epochs

net.train<cross_entropy>(optimizer, train_images, train_labels, batch_size, num_epochs);

net.test(test_images, test_labels).print_detail(std::cout);

net.save("lenet-model");
