#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

// Class for representing a node in the red-black tree
class RBNode {
public:
    int value;
    char color;  // 'R' or 'B'
    Mat image;
    RBNode* left, * right, * parent;
    RBNode(int value, char color, Mat image, RBNode* left = nullptr, RBNode* right = nullptr, RBNode* parent = nullptr) :
        value(value),color(color), image(image), left(left), right(right), parent(parent) {}
};

// Class for representing the red-black tree
class RedBlackTree {
public:
    RBNode* root;
    int size = 0;
    RedBlackTree(RBNode* root = nullptr) : root(root) {}

    // Method for inserting a new node into the tree
    void insert(int value,  char color, Mat image) {
        RBNode* new_node = new RBNode(value, color, image);
        if (root == nullptr) {
            root = new_node;
            new_node->color = 'B';  // Root node is always black
            return;
        }
        // Find the correct place to insert the new node using the "growing" method
        RBNode* current_node = root;
        while (current_node) {
            if (value < current_node->value) {
                //cout << "small value" << endl;
                if (current_node->left == nullptr) {
                    current_node->left = new_node;
                    new_node->parent = current_node;
                    break;
                }
                else {
                    current_node = current_node->left;
                }
            }
            else {
                //cout << "big value" << endl;
                if (current_node->right == nullptr) {
                   // cout << "1" << endl;
                    current_node->right = new_node;
                    new_node->parent = current_node;
                    break;
                }
                else {
                    //cout << "2" << endl;
                   // cout << "current_node->right: " << current_node->right << endl;
                    current_node = current_node->right;
                }
            }
        }
        // Fix any violations of the red-black tree properties
        fixViolations(new_node);
        root->color = 'B';
    }



    // Method for finding the node with the closest color value to a given value
    RBNode* findClosest(int value) {
        RBNode* closest_node = nullptr;
        int closest_distance = INT_MAX;
        RBNode* current_node = root;
        while (current_node) {
            int distance = abs(value - current_node->value);
            if (distance < closest_distance) {
                closest_node = current_node;
                closest_distance = distance;
            }
            if (value < current_node->value) {
                current_node = current_node->left;
            }
            else {
                current_node = current_node->right;
            }
            //cout << "distance"<<distance << endl;
        }
        return closest_node;
    }

    // Method for fixing violations of the red-black tree properties
    void fixViolations(RBNode* node) {
       // cout << "fixViolations" << endl;
        // Case 1: node is the root node (which is always black)
        if (node->parent == nullptr) {
           // cout << "node is the root node" << endl;
            return;
        }
        // Case 2: node's parent is black (no violation)
        if (node->parent->color == 'B') {
            //cout << "fixViolations Case 2" << endl;
            return;
        }
        // Case 3: node's uncle is red (simply recolo flip)
        RBNode* uncle = getUncle(node);
        if (uncle && uncle->color == 'R') {
            //cout << "fixViolations Case 3" << endl;
            node->parent->color = 'B';
            uncle->color = 'B';
            RBNode* grandparent = getGrandparent(node);
            grandparent->color = 'R';
            fixViolations(grandparent);
            return;
        }
        // LR Case 4: node's uncle is black and node is a right child with a left-leaning parent
        if (isRightChild(node) && isLeftChild(node->parent)) {
            //cout << "fixViolations Case 4" << endl;
            rotateLeft(node->parent);
            rotateRight(node->parent->parent);
            node->color = 'B';
            if (node->parent->parent) {
                node->parent->parent->color = 'R';
            }
        }
        // RL Case 5: node's uncle is black and node is a left child with a right-leaning parent
        else if (isLeftChild(node) && isRightChild(node->parent)) {
            //cout << "fixViolations Case 5" << endl;
            rotateRight(node->parent);
            rotateLeft(node->parent->parent);
            node->color = 'B';
            if (node->parent->parent) {
                node->parent->parent->color = 'R';
            }
        }     
        // LL Case 6: node's uncle is black and node is a left child with a left-leaning parent
        if (isLeftChild(node) && isLeftChild(node->parent)) {     
            //cout << "fixViolations Case 6" << endl;
            rotateRight(node->parent->parent);  
            node->parent->color = 'B';
            if (node->parent->parent) {
                node->parent->parent->color = 'R';
            }
        }
        // RR Case 7: node's uncle is black and node is a right child with a right-leaning parent
        else if (isRightChild(node) && isRightChild(node->parent)) { 
            //cout << "fixViolations Case 7" << endl;
            rotateLeft(node->parent->parent);
            node->parent->color = 'B';
            if (node->parent->parent) {
                node->parent->parent->color = 'R';
            }
        }
    }

    
    // Method for getting the uncle of a node
    RBNode* getUncle(RBNode* node) {
        if (node->parent == nullptr || node->parent->parent == nullptr) {
            return nullptr;
        }
        if (node->parent->parent->left == node->parent) {
            return node->parent->parent->right;
        }
        else {
            return node->parent->parent->left;
        }
    }

    // Method for checking if a node is a right child
    bool isRightChild(RBNode* node) {
        if (node->parent == nullptr) {
           // cout << "isRightChild false" << endl;
            return false;
        }
        return node->parent->right == node;
    }

    // Method for checking if a node is a left child
    bool isLeftChild(RBNode* node) {
        if (node->parent == nullptr) {
            //cout << "isLeftChild false" << endl;
            return false;
        }
        return node->parent->left == node;
    } 

    // Gets the grandparent of a node
    RBNode* getGrandparent(RBNode* node) {
        if (node == nullptr || node->parent == nullptr) {
            return nullptr;
        }
        return node->parent->parent;
    }

    // Method for performing a left rotation
    void rotateRight(RBNode* node) {
        // Perform the left rotation
        if (node== nullptr)return;
        RBNode* left_child = node->left;
        if (left_child == nullptr) {
            return;
        }

        RBNode* left_right_child = left_child->right;
        node->left = left_right_child;
        if (left_right_child != nullptr) {
            left_right_child->parent = node;
        }

        RBNode* node_parent = node->parent;
        left_child->parent = node_parent;
        if (node_parent == nullptr) {
            root = left_child;
        }
        else if (node_parent->left == node) {
            node_parent->left = left_child;
        }
        else {
            node_parent->right = left_child;
        }
        left_child->right = node;
        node->parent = left_child;
    }

    // Method for performing a right rotation
    void rotateLeft(RBNode* node) {
        if (node == nullptr)return;
        RBNode* right_child = node->right;
        if (right_child == nullptr) return;  // node->right is null, cannot rotate
        node->right = right_child->left;
        if (right_child->left != nullptr) right_child->left->parent = node;

        right_child->parent = node->parent;
        if (node->parent == nullptr) root = right_child;
        else if (isLeftChild(node)) node->parent->left = right_child;
        else node->parent->right = right_child;

        right_child->left = node;
        node->parent = right_child;
    }

    // Method for flipping the colors of a node and its children
    void colorFlip(RBNode* node) {
        node->color = (node->color == 'B') ? 'R' : 'B';
        if (node->left != nullptr) {
            node->left->color = (node->left->color == 'B') ? 'R' : 'B';
        }
        if (node->right != nullptr) {
            node->right->color = (node->right->color == 'B') ? 'R' : 'B';
        }
    }
};

// Function for creating a photomosaic image using the "divide and conquer" method
Mat createPhotomosaic(Mat target_image, RedBlackTree tree, int tile_size) {
    // Create a mosaic image with the same size as the target image
    Mat mosaic_image = Mat::zeros(target_image.rows, target_image.cols, CV_8UC3);

    // Divide the target image into a grid of tiles
    #pragma omp parallel for
    for (int y = 0; y < target_image.rows; y += tile_size) {
        for (int x = 0; x < target_image.cols; x += tile_size) {
            // Extract the current tile from the target image
            Mat tile = target_image(Rect(x, y, tile_size, tile_size));
            Scalar tile_mean = mean(tile);
            int color_value = (int)(0.3 * tile_mean[0] + 0.3 * tile_mean[1] + 0.3 * tile_mean[2]);
            // Find the closest matching tile image in the tree
            RBNode* closest_node = tree.findClosest(color_value);
            Mat closest_image = closest_node->image;

            // Replace the tile in the mosaic image with the closest matching tile image
            resize(closest_image, closest_image, cv::Size(tile_size, tile_size), INTER_AREA);
                    
            closest_image.copyTo(mosaic_image(Rect(x, y, tile_size, tile_size)));
            //for (int i = 0; i < tile_size; ++i) {
                //for (int j = 0; j < tile_size; ++j) {
                    //mosaic_image.at<Vec3b>(y + i, x + j) = closest_image.at<Vec3b>(i, j);//* 0.5+ target_image.at<Vec3b>(y + i, x + j) *0.5;
                //}
            //}          
        }
    }
    return mosaic_image;
}

int main() {
    // Load the target image and create the red-black tree
    double ts = (double)getTickCount();
    Mat target_image = imread("../4.jpg");
    //cvtColor(target_image, target_image, COLOR_BGR2HSV);
    RedBlackTree tree;
    // Use the "growing" method to gradually add reference images to the tree
    for (int i = 1; i <= 20000; i++) {
        // Load the reference image and compute its average color
        Mat reference_image = imread("../small_images/" + to_string(i) + ".jpg");
        //cvtColor(reference_image, reference_image, COLOR_BGR2HSV);
        Scalar reference_mean = mean(reference_image);
        int color_value = (int)(0.1 * reference_mean[0] + 0.3 * reference_mean[1] + 0.6 * reference_mean[2]);
        // Insert the reference image and its average color into the tree
        tree.insert(color_value, 'R', reference_image);
    }
    // Use the "divide and conquer" method to create the photomosaic image
    int tile_size = 10;
    Mat mosaic_image  = createPhotomosaic(target_image, tree, tile_size);
    double te = (double)getTickCount();
    double T = (te - ts) * 1000 / getTickFrequency();//µ¥Î»ms
    cout << "time: " << T << endl;
    // Save the mosaic image to a file
    imwrite("../mosaic_image_rb.jpg", mosaic_image);
    imshow("win", mosaic_image);
    waitKey(0);
    return 0;
}


