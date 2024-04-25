#include "utils.h"

RGB HSVtoRGB(double h, double s, double v)
{
    double r, g, b;
    int i = std::floor(h * 6);
    double f = h * 6 - i;
    double p = v * (1 - s);
    double q = v * (1 - f * s);
    double t = v * (1 - (1 - f) * s);

    switch (i % 6)
    {
    case 0:
        r = v, g = t, b = p;
        break;
    case 1:
        r = q, g = v, b = p;
        break;
    case 2:
        r = p, g = v, b = t;
        break;
    case 3:
        r = p, g = q, b = v;
        break;
    case 4:
        r = t, g = p, b = v;
        break;
    case 5:
        r = v, g = p, b = q;
        break;
    }

    return {static_cast<int>(r * 255), static_cast<int>(g * 255), static_cast<int>(b * 255)};
}

RGB valueToColor(double value, double min, double max)
{

    double normalized = std::min(std::max((value - min) / (max - min), 0.0), 1.0);

    double hue = (1.0 - normalized) * 240.0 / 360.0;

    return HSVtoRGB(hue, 1.0, 1.0);
}

Eigen::Vector3d rotate2rpy(Eigen::Matrix3d &rot)
{
    double roll = std::atan2(rot(2, 1), rot(2, 2));
    double pitch = asin(-rot(2, 0));
    double yaw = std::atan2(rot(1, 0), rot(0, 0));
    return Eigen::Vector3d(roll, pitch, yaw);
}