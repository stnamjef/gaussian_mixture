#pragma once
#include <iostream>
#include <cmath>
#include <Eigen/dense>
using namespace std;
using namespace Eigen;

#define PI 3.14159265358979323846

namespace stats
{
	RowVectorXd mean(const MatrixXd& dataset, int axis, const VectorXd& weight = {});
	void mean_row_operation(const MatrixXd& dataset, RowVectorXd& mu);
	void mean_col_operation(const MatrixXd& dataset, RowVectorXd& mu);
	void weighted_mean_row_operation(const MatrixXd& dataset, RowVectorXd& mu, const VectorXd& weight);
	void weighted_mean_col_operation(const MatrixXd& dataset, RowVectorXd& mu, const VectorXd& weight);

	double var(const RowVectorXd& vec1, const RowVectorXd& vec2, double mu1, double mu2,
		const VectorXd& weight = {});

	MatrixXd cov(const MatrixXd& dataset, const VectorXd& weight = {});

	double multivariate_normal(const RowVectorXd& X, const RowVectorXd& mu, const MatrixXd& sigma);


	RowVectorXd mean(const MatrixXd& dataset, int axis, const VectorXd& weight)
	{
		RowVectorXd mu;
		bool isEmpty = (weight.size() == 0);
		if (dataset.rows() == 0 || dataset.cols() == 0)
		{
			cout << "Error(mean(const MatrixXd&, RowVectorXd&, int)): Empty dataset." << endl;
			return mu;
		}

		if ((axis == 0) && isEmpty)
			mean_row_operation(dataset, mu);
		else if ((axis == 0) && !isEmpty)
			weighted_mean_row_operation(dataset, mu, weight);
		else if ((axis == 1) && isEmpty)
			mean_col_operation(dataset, mu);
		else
			weighted_mean_col_operation(dataset, mu, weight);
		return mu;
	}

	void mean_row_operation(const MatrixXd& dataset, RowVectorXd& mu)
	{
		Eigen::Index row = dataset.rows(), col = dataset.cols();

		mu.resize(row);

		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				if (j == 0)
					mu[i] = dataset(i, j);
				else
					mu[i] += dataset(i, j);
			}
			mu[i] /= (double)col;
		}
	}

	void mean_col_operation(const MatrixXd& dataset, RowVectorXd& mu)
	{
		Eigen::Index row = dataset.rows(), col = dataset.cols();

		mu.resize(col);

		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
			{
				if (i == 0)
					mu[j] = dataset(i, j);
				else
					mu[j] += dataset(i, j);

				if (i == row - 1)
					mu[j] /= (double)row;
			}
	}

	void weighted_mean_row_operation(const MatrixXd& dataset, RowVectorXd& mu, const VectorXd& weight)
	{
		if (dataset.cols() != weight.size())
		{
			cout << "Error(mean(const MatrixXd&, RowVectorXd&, int)): Invalid weight vector." << endl;
			return;
		}

		Eigen::Index row = dataset.rows(), col = dataset.cols();

		mu.resize(row);

		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
			{
				if (j == 0)
					mu[i] = weight[j] * dataset(i, j);
				else
					mu[i] += weight[j] * dataset(i, j);
			}
	}

	void weighted_mean_col_operation(const MatrixXd& dataset, RowVectorXd& mu, const VectorXd& weight)
	{
		if (dataset.rows() != weight.size())
		{
			cout << "Error(mean(const MatrixXd&, RowVectorXd&, int)): Invalid weight vector." << endl;
			return;
		}

		Eigen::Index row = dataset.rows(), col = dataset.cols();

		mu.resize(col);

		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
			{
				if (i == 0)
					mu[j] = weight[i] * dataset(i, j);
				else
					mu[j] += weight[i] * dataset(i, j);
			}
	}

	double var(const RowVectorXd& vec1, const RowVectorXd& vec2, double mu1, double mu2,
		const VectorXd& weight)
	{
		bool isEmpty = (weight.size() == 0);
		if (vec1.size() != vec2.size())
		{
			cout << "Error(var(const Block&, const Block&, double, double, const Block&)): " <<
				"Vectors are not compatible." << endl;
			return 0.0;
		}
		else if (!isEmpty && (vec1.size() != weight.size()))
		{
			cout << "Error(var(const Block&, const Block&, double, double, const Block&)): " <<
				"Invalid weight vector." << endl;
			return 0.0;
		}

		double dev = 0.0;
		for (int i = 0; i < vec1.size(); i++)
		{
			if (isEmpty)
				dev += (vec1[i] - mu1) * (vec2[i] - mu2);
			else
				dev += weight[i] * (vec1[i] - mu1) * (vec2[i] - mu2);
		}

		if (isEmpty)
			return dev / (vec1.size() - 1); // df = N - 1;
		else
			return dev;
	}

	MatrixXd cov(const MatrixXd& dataset, const VectorXd& weight)
		// WARNING
		//	- A dataset must not include a class vector.
		//	- A dataset must be a transposed matrix: (nFeature, nData)
	{
		MatrixXd sigma;
		bool isEmpty = (weight.size() == 0);
		if (dataset.rows() == 0 || dataset.cols() == 0)
		{
			cout << "Error(cov(const MatrixXd&, const MatrixXd::RowXpr*)): Empty dataset." << endl;
			return sigma;
		}

		RowVectorXd mu = mean(dataset, 0);

		Eigen::Index size = dataset.rows();
		sigma.resize(size, size);

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
			{
				if (i <= j)
					sigma(i, j) = var(dataset.row(i), dataset.row(j), mu[i], mu[j], weight);
				else
					sigma(i, j) = sigma(j, i);
			}

		return sigma;
	}

	double multivariate_normal(const RowVectorXd& X, const RowVectorXd& mu, const MatrixXd& sigma)
	{
		double phi, norm, quad;
		phi = std::pow((2 * PI), (X.cols() / 2.0));
		norm = std::pow(sigma.determinant(), 0.5);
		quad = (X - mu) * sigma.inverse() * (X - mu).transpose();
		quad = std::exp(quad / -2.0);

		return quad / (phi * norm);
	}
}