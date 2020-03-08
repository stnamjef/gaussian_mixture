#pragma once
#include <iostream>
#include <ctime>
#include <Eigen/Dense>
#include "stats.h"
#include "k_means.h"
using namespace std;

class GMM
{
private:
	int K;
	MatrixXd weights;
	RowVectorXd phi;
	RowVectorXd* MUs;
	MatrixXd* sigmas;
public:
	GMM(int K);
	~GMM();
	void fit(const MatrixXd& X, int init = 0);
	VectorXd predict(const MatrixXd& X);
};

namespace gmm
{
	void initialize(const MatrixXd& X, MatrixXd& weights, RowVectorXd& phi, RowVectorXd* MUs,
		MatrixXd* sigmas, int K, int init);

	void init_weights(MatrixXd& weights, int N, int K);

	void rand_init_phi(RowVectorXd& phi, int K);

	void rand_init_mu(const MatrixXd& X, RowVectorXd*& MUs, int K);

	int unique_random(const vector<int>& unique, int range);

	void rand_init_sigma(const MatrixXd& X, MatrixXd*& sigmas, int K);

	void kmeans_init_phi(const MatrixXd& X, RowVectorXd& phi, int K, const vector<vector<int>>& clusters);

	void kmeans_init_mu(RowVectorXd*& MUs, int K, KMeans& km);

	void kmeans_init_sigma(const MatrixXd& X, MatrixXd*& sigmas, int K, const vector<vector<int>>& clusters);

	void e_step(const MatrixXd& X, MatrixXd& weights, const RowVectorXd& phi, const RowVectorXd* MUs,
		const MatrixXd* sigmas, int K);

	void m_step(const MatrixXd& X, const MatrixXd& weights, RowVectorXd& phi, RowVectorXd*& MUs,
		MatrixXd*& sigmas, int K);

	double log_likelihood(const MatrixXd& X, const RowVectorXd& phi, const RowVectorXd* MUs,
		const MatrixXd* sigmas, int K);

	double likelihood(const MatrixXd& X, const RowVectorXd& phi, const RowVectorXd* MUs,
		const MatrixXd* sigmas, int K);

	void print_result(const RowVectorXd& phi, const RowVectorXd* MUs, const MatrixXd* sigmas, int K);
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

void GMM::fit(const MatrixXd& X, int init)
{
	using namespace gmm;

	if (init != 0 && init != 1)
	{
	cout << "Error(GMM::fit(const MatrixXd&, int)): Invalid initialization." << endl;
	return;
	}

	initialize(X, weights, phi, MUs, sigmas, K, init);

	int epoch = 1;
	double bef = 0.0, cur;
	while (1)
	{
		e_step(X, weights, phi, MUs, sigmas, K);
		m_step(X, weights, phi, MUs, sigmas, K);

		cur = log_likelihood(X, phi, MUs, sigmas, K);

		cout << "[ Epoch " << epoch << " ]";
		cout << " -> log-likelihood : " << cur << endl;

		if ((bef != 0) && (std::fabs(cur - bef) < 0.1))
			break;

		bef = cur;

		if (epoch > 100)
		{
			cout << "Warning(GMM::fit(const MatrixXd&, int)): over 100 iterations." << endl;
			break;
		}

		epoch++;
	}
	print_result(phi, MUs, sigmas, K);
}

void gmm::initialize(const MatrixXd& X, MatrixXd& weights, RowVectorXd& phi, RowVectorXd* MUs,
	MatrixXd* sigmas, int K, int init)
{
	init_weights(weights, (int)X.rows(), K);

	if (init == 0)
	{
		rand_init_phi(phi, K);
		rand_init_mu(X, MUs, K);
		rand_init_sigma(X, sigmas, K);
	}
	else
	{
		KMeans km(K);
		km.fit(X, 1, 5);

		vector<vector<int>> clusters = km.getClusters();

		kmeans_init_phi(X, phi, K, clusters);
		kmeans_init_mu(MUs, K, km);
		kmeans_init_sigma(X, sigmas, K, clusters);
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

void gmm::rand_init_mu(const MatrixXd& X, RowVectorXd*& MUs, int K)
{
	srand((unsigned)time(NULL));

	vector<int> unique;
	for (int i = 0; i < K; i++)
	{
		int idx = unique_random(unique, (int)X.rows());
		MUs[i] = X.row(idx);
		unique.push_back(idx);
	}
}

int gmm::unique_random(const vector<int>& unique, int range)
{
	bool isOverlap;
	int num;
	do
	{
		num = rand() % range;
		isOverlap = false;
		for (int i = 0; i < unique.size(); i++)
			if (unique[i] == num)
			{
				isOverlap = true;
				break;
			}
	} while (isOverlap);
	return num;
}

void gmm::rand_init_sigma(const MatrixXd& X, MatrixXd*& sigmas, int K)
{
	MatrixXd sigma = stats::cov(X.transpose());

	for (int i = 0; i < K; i++)
		sigmas[i] = sigma;
}

void gmm::kmeans_init_phi(const MatrixXd& X, RowVectorXd& phi, int K, const vector<vector<int>>& clusters)
{
	phi.resize(K);
	for (int i = 0; i < K; i++)
		phi[i] = clusters[i].size() / (double)X.rows();
}

void gmm::kmeans_init_mu(RowVectorXd*& MUs, int K, KMeans& km)
{
	for (int i = 0; i < K; i++)
		MUs[i] = km.getCenter(i);
}

void gmm::kmeans_init_sigma(const MatrixXd& X, MatrixXd*& sigmas, int K, const vector<vector<int>>& clusters)
{
	for (int i = 0; i < clusters.size(); i++)
	{
		MatrixXd feature(clusters[i].size(), X.cols());
		for (int j = 0; j < clusters[i].size(); j++)
			feature.row(j) = X.row(clusters[i][j]);
		sigmas[i] = stats::cov(feature.transpose());
	}
}

void gmm::e_step(const MatrixXd& X, MatrixXd& weights, const RowVectorXd& phi, const RowVectorXd* MUs,
	const MatrixXd* sigmas, int K)
{
	for (int i = 0; i < X.rows(); i++)
		for (int j = 0; j < K; j++)
		{
			double weight = phi[j] * stats::multivariate_normal(X.row(i), MUs[j], sigmas[j]);
			double total = weight;
			for (int p = 0; p < K; p++)
			{
				if (p == j)
					continue;
				total += phi[p] * stats::multivariate_normal(X.row(i), MUs[p], sigmas[p]);
			}
			weights(i, j) = weight / total;
		}
}

void gmm::m_step(const MatrixXd& X, const MatrixXd& weights, RowVectorXd& phi, RowVectorXd*& MUs,
	MatrixXd*& sigmas, int K)
{
	int N = (int)X.rows();
	for (int j = 0; j < K; j++)
	{
		double N_j = 0;
		for (int i = 0; i < N; i++)
			N_j += weights(i, j);

		MUs[j] = stats::mean(X, 1, weights.col(j)) / N_j;
		sigmas[j] = stats::cov(X.transpose(), weights.col(j)) / N_j;
		phi[j] = N_j / N;
	}
}

double gmm::log_likelihood(const MatrixXd& X, const RowVectorXd& phi, const RowVectorXd* MUs,
	const MatrixXd* sigmas, int K)
{
	double llkd = 0.0;
	for (int i = 0; i < X.rows(); i++)
	{
		double lkd = 0.0;
		for (int j = 0; j < K; j++)
			lkd += phi[j] * stats::multivariate_normal(X.row(i), MUs[j], sigmas[j]);
		llkd += std::log(lkd);
	}
	return llkd;
}

double gmm::likelihood(const MatrixXd& X, const RowVectorXd& phi, const RowVectorXd* MUs,
	const MatrixXd* sigmas, int K)
{
	double lkd = 0.0;
	for (int i = 0; i < X.rows(); i++)
	{
		double temp = 0.0;
		for (int j = 0; j < K; j++)
			temp += phi[j] * stats::multivariate_normal(X.row(i), MUs[j], sigmas[j]);
		lkd *= temp;
	}
	return lkd;
}

void gmm::print_result(const RowVectorXd& phi, const RowVectorXd* MUs, const MatrixXd* sigmas, int K)
{
	cout << "------------------------------------------" << endl;
	cout << "[ Model fit result ]" << endl << endl;
	cout << "Phi:" << endl;
	cout << phi << endl << endl;
	cout << "Mu:" << endl;
	for (int i = 0; i < K; i++)
		cout << MUs[i] << endl << endl;
	cout << endl;
	cout << "Sigma:" << endl;
	for (int i = 0; i < K; i++)
		cout << sigmas[i] << endl << endl;
	cout << endl;
}

VectorXd GMM::predict(const MatrixXd& X)
{
	VectorXd labels(X.rows());
	if (phi.size() == 0)
	{
		cout << "Error(GMM::predict(const MatrixXd&)): GMM is not fitted." << endl;
		return labels;
	}

	for (int i = 0; i < X.rows(); i++)
	{
		vector<double> temp_weights;
		for (int j = 0; j < K; j++)
			temp_weights.push_back(stats::multivariate_normal(X.row(i), MUs[j], sigmas[j]));
		auto max = std::max_element(temp_weights.begin(), temp_weights.end());
		labels[i] = (double)std::distance(temp_weights.begin(), max);
	}
	return labels;
}