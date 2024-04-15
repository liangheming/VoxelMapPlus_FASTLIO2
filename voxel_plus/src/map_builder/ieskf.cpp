#include "ieskf.h"

namespace kf
{

    Eigen::Matrix3d rightJacobian(const Eigen::Vector3d &inp)
    {
        return Sophus::SO3d::leftJacobian(inp).transpose();
    }

    void State::operator+=(const Vector23d &delta)
    {
        pos += delta.segment<3>(0);
        rot *= Sophus::SO3d::exp(delta.segment<3>(3)).matrix();
        rot_ext *= Sophus::SO3d::exp(delta.segment<3>(6)).matrix();
        pos_ext += delta.segment<3>(9);
        vel += delta.segment<3>(12);
        bg += delta.segment<3>(15);
        ba += delta.segment<3>(18);
        g = Sophus::SO3d::exp(getBx() * delta.segment<2>(21)).matrix() * g;
    }

    void State::operator+=(const Vector24d &delta)
    {
        pos += delta.segment<3>(0);
        rot *= Sophus::SO3d::exp(delta.segment<3>(3)).matrix();
        rot_ext *= Sophus::SO3d::exp(delta.segment<3>(6)).matrix();
        pos_ext += delta.segment<3>(9);
        vel += delta.segment<3>(12);
        bg += delta.segment<3>(15);
        ba += delta.segment<3>(18);
        g = Sophus::SO3d::exp(delta.segment<3>(21)).matrix() * g;
    }

    Vector23d State::operator-(const State &other)
    {
        Vector23d delta = Vector23d::Zero();
        delta.segment<3>(0) = pos - other.pos;
        delta.segment<3>(3) = Sophus::SO3d(other.rot.transpose() * rot).log();
        delta.segment<3>(6) = Sophus::SO3d(other.rot_ext.transpose() * rot_ext).log();
        delta.segment<3>(9) = pos_ext - other.pos_ext;
        delta.segment<3>(12) = vel - other.vel;
        delta.segment<3>(15) = bg - other.bg;
        delta.segment<3>(18) = ba - other.ba;

        double v_sin = (Sophus::SO3d::hat(g) * other.g).norm();
        double v_cos = g.transpose() * other.g;
        double theta = std::atan2(v_sin, v_cos);
        Eigen::Vector2d res;
        if (v_sin < 1e-11)
        {
            if (std::fabs(theta) > 1e-11)
            {
                res << 3.1415926, 0;
            }
            else
            {
                res << 0, 0;
            }
        }
        else
        {
            res = theta / v_sin * other.getBx().transpose() * Sophus::SO3d::hat(other.g) * g;
        }
        delta.segment<2>(21) = res;
        return delta;
    }

    Matrix3x2d State::getBx() const
    {
        Matrix3x2d res;
        res << -g[1], -g[2],
            GRAVITY - g[1] * g[1] / (GRAVITY + g[0]), -g[2] * g[1] / (GRAVITY + g[0]),
            -g[2] * g[1] / (GRAVITY + g[0]), GRAVITY - g[2] * g[2] / (GRAVITY + g[0]);
        res /= GRAVITY;
        return res;
    }

    Matrix3x2d State::getMx() const
    {

        return -Sophus::SO3d::hat(g) * getBx();
    }

    Matrix3x2d State::getMx(const Eigen::Vector2d &res) const
    {
        Matrix3x2d bx = getBx();
        Eigen::Vector3d bu = bx * res;
        return -Sophus::SO3d::exp(bu).matrix() * Sophus::SO3d::hat(g) * Sophus::SO3d::leftJacobian(bu).transpose() * bx;
    }

    Matrix2x3d State::getNx() const
    {
        return 1 / GRAVITY / GRAVITY * getBx().transpose() * Sophus::SO3d::hat(g);
    }

    IESKF::IESKF() = default;

    int IESKF::P_ID = 0, IESKF::R_ID = 3, IESKF::ER_ID = 6, IESKF::EP_ID = 9, IESKF::V_ID = 12, IESKF::BG_ID = 15, IESKF::BA_ID = 18, IESKF::G_ID = 21;

    void IESKF::predict(const Input &inp, double dt, const Matrix12d &Q)
    {
        Vector24d delta = Vector24d::Zero();
        delta.segment<3>(0) = x_.vel * dt;
        delta.segment<3>(3) = (inp.gyro - x_.bg) * dt;
        delta.segment<3>(12) = (x_.rot * (inp.acc - x_.ba) + x_.g) * dt;
        F_.setIdentity();
        F_.block<3, 3>(0, 12) = Eigen::Matrix3d::Identity() * dt;
        F_.block<3, 3>(3, 3) = Sophus::SO3d::exp(-(inp.gyro - x_.bg) * dt).matrix();
        F_.block<3, 3>(3, 15) = -rightJacobian((inp.gyro - x_.bg) * dt) * dt;
        F_.block<3, 3>(12, 3) = -x_.rot * Sophus::SO3d::hat(inp.acc - x_.ba) * dt;
        F_.block<3, 3>(12, 18) = -x_.rot * dt;
        F_.block<3, 2>(12, 21) = x_.getMx() * dt;
        F_.block<2, 2>(21, 21) = x_.getNx() * x_.getMx();

        G_.setZero();
        G_.block<3, 3>(3, 0) = -rightJacobian((inp.gyro - x_.bg) * dt) * dt;
        G_.block<3, 3>(12, 3) = -x_.rot * dt;
        G_.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity() * dt;
        G_.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity() * dt;
        x_ += delta;
        P_ = F_ * P_ * F_.transpose() + G_ * Q * G_.transpose();
    }

    void IESKF::update()
    {
        State predict_x = x_;
        SharedState shared_data;
        shared_data.iter_num = 0;
        Vector23d delta = Vector23d::Zero();
        for (size_t i = 0; i < max_iter_; i++)
        {
            func_(x_, shared_data);
            H_.setZero();
            b_.setZero();
            delta = x_ - predict_x;
            Matrix23d J = Matrix23d::Identity();
            J.block<3, 3>(3, 3) = rightJacobian(delta.segment<3>(3));
            J.block<3, 3>(6, 6) = rightJacobian(delta.segment<3>(6));
            J.block<2, 2>(21, 21) = x_.getNx() * predict_x.getMx(delta.segment<2>(21));
            b_ += (J.transpose() * P_.inverse() * delta);
            H_ += (J.transpose() * P_.inverse() * J);
            H_.block<12, 12>(0, 0) += shared_data.H;
            b_.block<12, 1>(0, 0) += shared_data.b;
            delta = -H_.inverse() * b_;
            x_ += delta;
            shared_data.iter_num += 1;
            if (delta.maxCoeff() < eps_)
                break;
        }
        Matrix23d L = Matrix23d::Identity();
        L.block<3, 3>(3, 3) = rightJacobian(delta.segment<3>(3));
        L.block<3, 3>(6, 6) = rightJacobian(delta.segment<3>(6));
        L.block<2, 2>(21, 21) = x_.getNx() * predict_x.getMx(delta.segment<2>(21));
        P_ = L * H_.inverse() * L.transpose();
    }

} // namespace kf