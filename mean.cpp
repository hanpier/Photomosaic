#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Structure to store a single tile image
struct Tile {
    Mat image;  // The image data
    int r, g, b;  // The average color of the image
};

int main() {
    // Load the target image
    double ts = (double)getTickCount();
    Mat target_image = imread("../4.jpg");

    // Load the tile images
    vector<Tile> tiles;
    for (int i = 1; i < 20000; i++) {
        Mat tile_image = imread("../small_images/" + to_string(i) + ".jpg");
        if (tile_image.empty()) {
            break;
        }

        // Calculate the average color of the tile image
        int r = 0, g = 0, b = 0;
        for (int y = 0; y < tile_image.rows; y++) {
            for (int x = 0; x < tile_image.cols; x++) {
                Vec3b color = tile_image.at<Vec3b>(y, x);
                b += color[0];
                g += color[1];
                r += color[2];
            }
        }
        r /= tile_image.rows * tile_image.cols;
        g /= tile_image.rows * tile_image.cols;
        b /= tile_image.rows * tile_image.cols;

        // Add the tile to the vector
        tiles.push_back({ tile_image, r, g, b });
    }

    // Create a mosaic image with the same size as the target image
    Mat mosaic_image(target_image.rows, target_image.cols, CV_8UC3, Scalar(0, 0, 0));

    // Divide the target image into small regions
    int tile_size = 10;  // The size of each tile in the mosaic
    #pragma omp parallel for
    for (int y = 0; y < target_image.rows; y += tile_size) {
        for (int x = 0; x < target_image.cols; x += tile_size) {
            // Calculate the average color of the region
            int r = 0, g = 0, b = 0;
            for (int i = y; i < y + tile_size && i < target_image.rows; i++) {
                for (int j = x; j < x + tile_size && j < target_image.cols; j++) {
                    Vec3b color = target_image.at<Vec3b>(i, j);
                    b += color[0];
                    g += color[1];
                    r += color[2];
                }
            }
            r /= tile_size * tile_size;
            g /= tile_size * tile_size;
            b /= tile_size * tile_size;


            // Find the tile with the closest average color
            int best_tile = -1;
            int min_distance = INT_MAX;
            for (int i = 0; i < tiles.size(); i++) {
                int distance = (tiles[i].r - r)* (tiles[i].r - r) + (tiles[i].g - g)* (tiles[i].g - g) + (tiles[i].b - b)* (tiles[i].b - b);
                if (distance < min_distance) {
                    best_tile = i;
                    min_distance = distance;
                }
            }
           // cout << "best_tile: " << best_tile << endl;
            //Mat resized;
            //imshow("wins", tiles[best_tile].image);
            //waitKey(0);
            resize(tiles[best_tile].image, tiles[best_tile].image, Size(tile_size, tile_size));
            // Paste the best tile into the mosaic image
            //Mat roi(mosaic_image, Rect(x, y, tile_size, tile_size));
            tiles[best_tile].image.copyTo(mosaic_image(Rect(x, y, tile_size, tile_size)));
        }
    }
    double te = (double)getTickCount();
    double T = (te - ts) * 1000 / getTickFrequency();//µ¥Î»ms
    cout << "time: "<< T << endl;
    // Save the mosaic image
    imshow("win", mosaic_image);
    waitKey(0);
    imwrite("../mosaic_image_mean.jpg", mosaic_image);

    return 0;
}
