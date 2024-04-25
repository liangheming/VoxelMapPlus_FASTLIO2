#pragma once
#include <cmath>
#include <algorithm>
#include <Eigen/Eigen>

struct RGB
{
    int r;
    int g;
    int b;
};

RGB HSVtoRGB(double h, double s, double v);

RGB valueToColor(double value, double min, double max);

Eigen::Vector3d rotate2rpy(Eigen::Matrix3d &rot);