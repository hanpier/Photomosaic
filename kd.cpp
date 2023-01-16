#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>

typedef int type;

using namespace std;
using namespace cv;

string input_folder = "../image";
string output_folder = "../small_images";

struct Parameters {
    string target_image_path;
    string reference_image_folder;
    string mosaic_image_path;
    int tile_size;
    int num_small;
    Parameters() : target_image_path(""), reference_image_folder(""), mosaic_image_path(""), tile_size(5), num_small(10000) {}
};

Parameters readParameters(const string& filepath) {
    Parameters parameters;

    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(filepath, pt);

    // Get parameters from file
    parameters.target_image_path = pt.get<string>("path.target_image");
    parameters.reference_image_folder = pt.get<string>("path.reference_image_folder");
    parameters.mosaic_image_path = pt.get<string>("path.mosaic_image");
    parameters.tile_size = pt.get<int>("parameter.tile_size");
    parameters.num_small = pt.get<int>("parameter.num_small");
    return parameters;
}

struct DataWithImg {
    vector<type> data;
    Mat img;
    DataWithImg(const vector<type>& data, const Mat& img) : data(data), img(img) {}
};

class KdNode {
private:
    DataWithImg  data_img;
    KdNode* left, * right;   
public:
    //KdNode(const vector<int>& data, const Mat& img) : data_img(DataWithImg(data, img)), left(nullptr), right(nullptr) {}
    KdNode(const DataWithImg& data_img) : data_img(data_img), left(nullptr), right(nullptr) {}
    ~KdNode() {
        if (left) delete left;
        if (right) delete right;
    }
    const vector<type>& getData() const { return data_img.data; }
    const Mat& getImg() const { return data_img.img; }
    KdNode*& buildLeft() { return left; }
    KdNode*& buildRight() { return right; }
    KdNode*  getLeft() { return left; }
    KdNode*  getRight() { return right; }
};

class KdTree {
private:
    KdNode* root;
    void build(vector<DataWithImg> data_img, KdNode*& node, int depth, int dim);
    double CalDistance(const vector<type>& a, const vector<type>& b, int dim);
    KdNode* findClosest(DataWithImg& data_img, KdNode* node, int depth, int dim);
    void release(KdNode* node);
public:
    KdTree() : root(nullptr) {}
    void build(vector<DataWithImg>data_img, int dim);
    KdNode* findClosest(DataWithImg& data, int dim);
    void release();
};
void KdTree::build(vector<DataWithImg> data_img, KdNode*& node, int depth, int dim) {
    if (data_img.empty()) return;
    // Select axis based on depth so that axis cycles through all valid values
    int axis = depth % dim;
    // Sort the data based on the selected axis and choose median as pivot element
    auto compare = [axis](const DataWithImg a, const DataWithImg b) { return a.data[axis] < b.data[axis]; };
    nth_element(data_img.begin(), data_img.begin() + data_img.size() / 2 , data_img.end(),compare);
    int medianIndex = data_img.size() / 2;
    vector<type> median = data_img[medianIndex].data;
    Mat medianImg = data_img[medianIndex].img;
    // Create new node and recursively construct subtrees
    DataWithImg median_str(median, medianImg);
    node = new KdNode(median_str);
    vector<DataWithImg> leftData, rightData;
    for (int i = 0; i < data_img.size(); i++) {
        if (i != medianIndex) {
            if (data_img[i].data[axis] < median[axis]) {
                leftData.push_back(data_img[i]);
            }
            else {
                rightData.push_back(data_img[i]);
            }
        }
    }
    build(leftData,node->buildLeft(), depth + 1, dim);
    build(rightData, node->buildRight(), depth + 1, dim);
}
void KdTree::build(vector<DataWithImg> data,int dim) {
    build(data, root, 0, dim);
}

double KdTree::CalDistance(const vector<type>& a, const vector<type>& b ,int dim) {
    double distance = 0;
    for (int i = 0; i < dim; i++) {
        distance += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(distance);
}

KdNode* KdTree::findClosest(DataWithImg& data_img, KdNode* node, int depth,int dim) {
    if (!node) {
        return nullptr;
    }
    int axis = depth % dim;
    KdNode* next = nullptr, * opposite = nullptr;
    if (data_img.data[axis] < node->getData()[axis]) {
        next = node->getLeft();
        opposite = node->getRight();
    }
    else {
        next = node->getRight();
        opposite = node->getLeft();
    }

    KdNode* best = findClosest(data_img, next, depth + 1, dim);
    if (!best) {
        best = node;
    }
    else {
        double best_distance = CalDistance(best->getData(), data_img.data, dim);
        double node_distance = CalDistance(node->getData(), data_img.data, dim);
        if (node_distance < best_distance) {
            best = node;
        }
    }
    double best_distance = CalDistance(best->getData(), data_img.data, dim);
    if (CalDistance(data_img.data,node->getData(), dim) > best_distance) {
        return best;
    }
    else {
        KdNode* other_best = findClosest(data_img, opposite, depth + 1, dim);
        if (!other_best) {
            return best;
        }
        else {
            double other_best_distance = CalDistance(other_best->getData(), data_img.data, dim);
            if (other_best_distance < best_distance) {
                return other_best;
            }
            else {
                return best;
            }
        }
    }
}
KdNode* KdTree::findClosest(DataWithImg& data_img,int dim) {
    return findClosest(data_img, root, 0,dim);
}

void KdTree::release(KdNode* node) {
    if (!node) {
        cout << "No node!" << endl;
        return;
    }       
    delete node;
}

void KdTree::release() {
    release(root);
    root = nullptr;
}

// Function for creating a photomosaic image using the "divide and conquer" method

Mat createPhotomosaic(Mat target_image, KdTree tree, int tile_size,int dim) {
    // Create a mosaic image with the same size as the target image
    Mat mosaic_image = Mat::zeros(target_image.rows, target_image.cols, CV_8UC3);
    // Divide the target image into a grid of tiles
    #pragma omp parallel for
    for (int y = 0; y < target_image.rows; y += tile_size) {
        for (int x = 0; x < target_image.cols; x += tile_size) {
            // Extract the current tile from the target image
            Mat tile = target_image(Rect(x, y, tile_size, tile_size));
            Scalar tile_mean = mean(tile);
            vector<type> rgb;
            rgb.push_back(tile_mean[0]);
            rgb.push_back(tile_mean[1]);
            rgb.push_back(tile_mean[2]);
            // Find the closest matching tile image in the tree
            DataWithImg data_img(rgb, tile);
            KdNode* closest_node = tree.findClosest(data_img, dim); 
            Mat closest_image = closest_node->getImg();
            // Replace the tile in the mosaic image with the closest matching tile image
            resize(closest_image, closest_image, cv::Size(tile_size, tile_size));
            closest_image.copyTo(mosaic_image(Rect(x, y, tile_size, tile_size)));
            //for (int i = 0; i < tile_size; ++i) {
                //for (int j = 0; j < tile_size; ++j) {
                   // mosaic_image.at<Vec3b>(y + i, x + j) = closest_image.at<Vec3b>(i, j);
               // }zh
           // }
        }
    }
    return mosaic_image;
}
void process_folder(const string& path) {
    // open folder
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        cerr << "Error opening folder: " << path << endl;
        return;
    }
    // Traverse files in folders
    struct dirent* entry;
    int i = 0;
    while ((entry = readdir(dir)) != nullptr) {
        string filename = entry->d_name;
        // Ignore '.' and '..'
        if (filename == "." || filename == "..") {
            continue;
        }

        // Build the full path of the input file
        string input_path = path + '/' + filename;
        // Get file information
        struct stat file_stat;
        stat(input_path.c_str(), &file_stat);
        // Determine whether the file is a folder
        if (S_ISDIR(file_stat.st_mode)) {
            // Recursively traverse files in folders
            process_folder(input_path);
        }
        else {
            // process the file
            Mat image_read = imread(input_path);
            // Save output file
            string output_path = output_folder + '/' + to_string(i + 1) + ".jpg";
            // Save file
            // ...
            imwrite(output_path, image_read);
            i++;
        }
    }

    // Close Folder
    closedir(dir);
}
int main() {
    //process_folder(input_folder);
    //Read parameter
    double ts = (double)getTickCount();
    int dim = 3;

    Parameters parameters = readParameters("../config.ini");
    // Load image
    Mat target_image = imread(parameters.target_image_path);
    // Create vector to store individual RGB values
    vector<DataWithImg> data_imgs;
    KdTree tree;
    for (int i = 1; i <= parameters.num_small; i++) {
        Mat reference_image = imread(parameters.reference_image_folder + '/' + to_string(i) + ".jpg");
        Scalar reference_mean = mean(reference_image);
        vector<type> rgb;
        rgb.push_back(reference_mean[0]);
        rgb.push_back(reference_mean[1]);
        rgb.push_back(reference_mean[2]);
        data_imgs.emplace_back(DataWithImg(rgb, reference_image));
    }   
    tree.build(data_imgs, dim);  
    Mat mosaic_image = createPhotomosaic(target_image, tree, parameters.tile_size,dim);
    double te = (double)getTickCount();
    double T = (te - ts) * 1000 / getTickFrequency();//µ¥Î»ms
    cout << "time: " << T << endl;
    //imshow("win", mosaic_image);
    //waitKey(0);
    imwrite(parameters.mosaic_image_path, mosaic_image);
    tree.release();
    return 0;
}
