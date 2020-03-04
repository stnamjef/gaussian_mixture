#pragma once
#include <iostream>
#include <ctime>
#include <Eigen/Dense>
#include "Stats.h"
#include "KMeans.h"
using namespace std;

class GMM
{
private:
	MatrixXd weights;
	RowVectorXd phi;
	RowVectorXd* MUs;
	MatrixXd* sigmas;
	int K;
public:
	GMM(int K);
	~GMM();
	void fit(const MatrixXd& df, int init = 0);
	VectorXd predict(const MatrixXd& df);
};

namespace gmm
{
	void initialize(const MatrixXd& df, MatrixXd& weights, RowVectorXd& phi,
		RowVectorXd* MUs, MatrixXd* sigmas, int K, int init);

	void init_weights(MatrixXd& weights, int N, int K);

	void rand_init_phi(RowVectorXd& phi, int K);

	void rand_init_mu(const MatrixXd& df, RowVectorXd*& MUs, int K);

	int unique_random(const int* unique, int size, int range);

	void rand_init_sigma(const MatrixXd& df, MatrixXd*& sigmas, int K);

	void kmeans_init_phi(const MatrixXd& df, RowVectorXd& phi, int K,
		const vector<vector<int>>& clusters);

	void kmeans_init_mu(const MatrixXd& df, RowVectorXd*& MUs, int K, KMeans& km);

	void kmeans_init_sigma(const MatrixXd& df, MatrixXd*& sigmas, int K,
		const vector<vector<int>>& clusters);

	void e_step(const MatrixXd& df, MatrixXd& weights, const RowVectorXd& phi,
		const RowVectorXd* MUs, const MatrixXd* sigmas, int K);

	void m_step(const MatrixXd& df, const MatrixXd& weights, RowVectorXd& phi,
		RowVectorXd*& MUs, MatrixXd*& sigmas, int K);

	double log_likelihood(const MatrixXd& df, const RowVectorXd& phi,
		const RowVectorXd* MUs, const MatrixXd* sigmas, int K);

	double likelihood(const MatrixXd& df, const RowVectorXd& phi,
		const RowVectorXd* MUs, const MatrixXd* sigmas, int K);
}

GMM::GMM(int K) : K(0), MUs(nullptr), sigmas(nullptr)
{
	if (K < 1)
		cout << "Error(GMM::GMM(int)): Invalid argument." << endl;
	else
	{
		this->K = K;
		MUs = new RowVectorXd[K];
		sigmas = new MatrixXd[K];
	}
}

GMM::~GMM()
{
	delete[] MUs;
	delete[] sigmas;
}

void GMM::fit(const MatrixXd& df, int init)
{
	using namespace gmm;

	if (df.rows() == 0 || df.cols() == 0)
	{
		cout << "Error(GMM::fit(const MatrixXd&, int)): Empty dataset." << endl;
		return;
	}
	else if (init != 0 && init != 1)
	{
		cout << "Error(GMM::fit(const MatrixXd&, int)): Invalid initialization." << endl;
		return;
	}

	initialize(df, weights, phi, MUs, sigmas, K, init);

	double bef = 0.0, cur;
	int n = 0;
	while (n < 10)
	{
		e_step(df, weights, phi, MUs, sigmas, K);
		m_step(df, weights, phi, MUs, sigmas, K);

		cur = log_likelihood(df, phi, MUs, sigmas, K);

		if (bef == 0.0)
			bef = cur;

		if (std::fabs(cur - bef) < 0.01)
			break;

		n++;
	}
}

void gmm::initialize(const MatrixXd& df, MatrixXd& weights, RowVectorXd& phi,
	RowVectorXd* MUs, MatrixXd* sigmas, int K, int init)
{
	init_weights(weights, (int)df.rows(), K);

	if (init == 0)
	{
		rand_init_phi(phi, K);
		rand_init_mu(df, MUs, K);
		rand_init_sigma(df, sigmas, K);
	}
	else
	{
		KMeans km(K);
		km.fit(df, 1, 5);

		vector<vector<int>> clusters;
		km.getClusters(clusters);

		kmeans_init_phi(df, phi, K, clusters);
		kmeans_init_mu(df, MUs, K, km);
		kmeans_init_sigma(df, sigmas, K, clusters);
	}
}

void gmm::init_weights(MatrixXd& weights, int N, int K)
// N: # of data
// K: # of normal distribution
{
	weights.resize(N, K);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < K; j++)
			weights(i, j) = 1 / (double)K;
}

void gmm::rand_init_phi(RowVectorXd& phi, int K)
{
	phi.resize(K);
	for (int i = 0; i < K; i++)
		phi[i] = 1 / (double)K;
}

void gmm::rand_init_mu(const MatrixXd& df, RowVectorXd*& MUs, int K)
{
	srand((unsigned)time(NULL));

	int* unique = new int[K];
	for (int i = 0; i < K; i++)
		MUs[i] = df.row(unique_random(unique, K, df.rows()));
}

int gmm::unique_random(const int* unique, int size, int range)
{
	bool isOverlap;
	int num;
	do
	{
		num = rand() % range;
		isOverlap = false;
		for (int i = 0; i < size; i++)
			if (unique[i] == num)
			{
				isOverlap = true;
				break;
			}
	} while (isOverlap);
	return num;
}

void gmm::rand_init_sigma(const MatrixXd& df, MatrixXd*& sigmas, int K)
{
	MatrixXd sigma = stats::cov(df.transpose());

	for (int i = 0; i < K; i++)
		sigmas[i] = sigma;
}

void gmm::kmeans_init_phi(const MatrixXd& df, RowVectorXd& phi, int K,
	const vector<vector<int>>& clusters)
{
	phi.resize(K);
	for (int i = 0; i < K; i++)
		phi[i] = clusters[i].size() / (double)df.rows();
}

void gmm::kmeans_init_mu(const MatrixXd& df, RowVectorXd*& MUs, int K, KMeans& km)
{
	for (int i = 0; i < K; i++)
		km.getCenters(MUs);
}

void gmm::kmeans_init_sigma(const MatrixXd& df, MatrixXd*& sigmas, int K,
	const vector<vector<int>>& clusters)
{
	for (int i = 0; i < clusters.size(); i++)
	{
		MatrixXd temp(clusters[i].size(), df.cols());
		for (int j = 0; j < clusters[i].size(); j++)
		{
			temp.row(j) = df.row(clusters[i][j]);
		}
		sigmas[i] = stats::cov(temp.transpose());
	}
}

void gmm::e_step(const MatrixXd& df, MatrixXd& weights, const RowVectorXd& phi,
	const RowVectorXd* MUs, const MatrixXd* sigmas, int K)
{
	for (int i = 0; i < df.rows(); i++)
		for (int j = 0; j < K; j++)
		{
			double weight = phi[j] * stats::multivariate_normal(df.row(i), MUs[j], sigmas[j]);
			double total = weight;
			for (int p = 0; p < K; p++)
			{
				if (p == j)
					continue;
				total += phi[p] * stats::multivariate_normal(df.row(i), MUs[p], sigmas[p]);
			}
			weights(i, j) = weight / total;
		}
}

void gmm::m_step(const MatrixXd& df, const MatrixXd& weights, RowVectorXd& phi,
	RowVectorXd*& MUs, MatrixXd*& sigmas, int K)
{
	for (int j = 0; j < K; j++)
	{
		double N_j = 0.0;
		for (int i = 0; i < df.rows(); i++)
			N_j += weights(i, j);

		MUs[j] = stats::mean(df, 1, weights.col(j)) / N_j;
		sigmas[j] = stats::cov(df.transpose(), weights.col(j)) / N_j;
		phi[j] = N_j / (double)df.rows();
	}
}

double gmm::log_likelihood(const MatrixXd& df, const RowVectorXd& phi,
	const RowVectorXd* MUs, const MatrixXd* sigmas, int K)
{
	double llkd = 0.0;
	for (int i = 0; i < df.rows(); i++)
	{
		double lkd = 0.0;
		for (int j = 0; j < K; j++)
			lkd += phi[j] * stats::multivariate_normal(df.row(i), MUs[j], sigmas[j]);
		llkd += std::log(lkd);
	}
	return llkd;
}

double gmm::likelihood(const MatrixXd& df, const RowVectorXd& phi,
	const RowVectorXd* MUs, const MatrixXd* sigmas, int K)
{
	double lkd = 0.0;
	for (int i = 0; i < df.rows(); i++)
	{
		double temp = 0.0;
		for (int j = 0; j < K; j++)
			temp += phi[j] * stats::multivariate_normal(df.row(i), MUs[j], sigmas[j]);
		lkd *= temp;
	}
	return lkd;
}

VectorXd GMM::predict(const MatrixXd& df)
{
	VectorXd labels(df.rows());
	if (phi.size() == 0)
	{
		cout << "Error(GMM::predict(const MatrixXd&)): GMM is not fitted." << endl;
		return labels;
	}

	MatrixXd temp_weight(df.rows(), K);
	gmm::e_step(df, temp_weight, phi, MUs, sigmas, K);

	for (int i = 0; i < df.rows(); i++)
	{
		int label = K - 1;
		double max = temp_weight(i, (K - 1));
		for (int j = 0; j < K - 1; j++)
		{
			if (max < temp_weight(i, j))
			{
				max = temp_weight(i, j);
				label = j;
			}
		}
		labels[i] = label;
	}

	vector<vector<int>> clusters;
	kms::to_clusters(labels, K, clusters);

	cout << "Silhouette score: " <<
		kms::silhouette_score(df, clusters, MUs, K) << endl;

	return labels;
}