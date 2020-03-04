#pragma once
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

/*
	K means clustering

		- This code is written to deeply understand the kmeans algorithm.

	1. Variable description

		- K: # of clusters
		- groups: A 2d vector containing the original dataset's index of each group's point.
		- centers: A row vector containing the center of each cluster.

	2. Function description

		- fit(const MatrixXd& df, int init, int n_init):

			This function takes three arguments and df is a dataset used to form clusters. init
			is an option to initialize centers. If it is zero, the function randomly select
			'K' number of rows from dataset and assign them as initail centers. If it is one,
			the function initialize centers using kmeans++ algorithm. This algorithm selects
			the furthest point from the existing centers. In gerneral, this is known to prevent
			a convergence to the local minimum, guaranteeing slightly better performance.

			The thrid argument represents the number of initializations. If it is one, the default
			value, the function forms one cluster set using only one center set. In other words,
			it starts at only one point(an initial set of centers) and end when it reaches
			the minimum(the optimal centers). If it is greater than one, the funtion forms multiple
			cluseter sets using mutiple center set, and selects the best one among them.

		- predict(const MatrixXd& df):

			This function simply assigns each point(each row of datset) to the closest cluster.
			It also calculate the sillhouette coefficient to evaluate the clustring result.
			The coefficient has a value between -1 to 1. The closer to 1, the better the clusters.
*/

class KMeans
{
private:
	int K;
	vector<vector<int>> groups;
	RowVectorXd* centers;
public:
	KMeans(int n_clusters);
	~KMeans();
	void fit(const MatrixXd& df, int init, int n_init = 1);
	VectorXd predict(const MatrixXd& df);
	void getCenters(RowVectorXd*& copy);
	void getClusters(vector<vector<int>>& copy);
};

namespace kms
{
	void single_fit(const MatrixXd& df, vector<vector<int>>& groups, RowVectorXd*& centers,
		int K, int init, int seef = 0);

	void rand_init_center(const MatrixXd& df, RowVectorXd*& centers, int K, int seed);

	int unique_random(const int* unique, int size, int range);

	void kmpp_init_center(const MatrixXd& df, RowVectorXd*& centers, int K, int seed);

	int nearest_center(const RowVectorXd& X, const RowVectorXd* centers, int K);

	double euclidean_norm(const RowVectorXd& p1, const RowVectorXd& p2);

	void assign_group(const MatrixXd& df, vector<vector<int>>& groups,
		const RowVectorXd* centers, int K);

	bool isSame(const vector<vector<int>>& before, const vector<vector<int>>& current);

	void update_center(const MatrixXd& df, const vector<vector<int>>& groups,
		RowVectorXd*& centers);

	void ensemble_fit(const MatrixXd& df, vector<vector<int>>& groups, RowVectorXd*& centers,
		int K, int init, int n_init);

	double squared_error(const MatrixXd& df, const vector<vector<int>>& groups,
		const RowVectorXd* centers);

	void to_clusters(const VectorXd& labels, int K, vector<vector<int>>& clusters);

	double silhouette_score(const MatrixXd& df, const vector<vector<int>>& groups,
		const RowVectorXd* centers, int K);

	double mean_distance(const MatrixXd& df, int idx, const vector<int>& group);
}

KMeans::KMeans(int n_clusters) : K(0), centers(nullptr)
{
	if (n_clusters <= 0)
		cout << "Error(KMeans::KMeans(int)): Invalid clusters." << endl;
	else
	{
		K = n_clusters;
		centers = new RowVectorXd[K];
	}
}

KMeans::~KMeans() { delete[] centers; }

void KMeans::fit(const MatrixXd& df, int init, int n_init)
{
	if (df.rows() == 0 || df.cols() == 0)
	{
		cout << "Error(mean(const MatrixXd&, RowVectorXd&, int)): Empty dataset." << endl;
		return;
	}

	if (n_init == 1)
		kms::single_fit(df, groups, centers, K, init);
	else if (n_init > 1)
		kms::ensemble_fit(df, groups, centers, K, init, n_init);
	else
		cout << "Error(mean(const MatrixXd&, RowVectorXd&, int)): Invalide initialization." << endl;
}

VectorXd KMeans::predict(const MatrixXd& df)
{
	VectorXd labels(df.rows());
	for (int i = 0; i < df.rows(); i++)
		labels[i] = kms::nearest_center(df.row(i), centers, K);

	vector<vector<int>> clusters;
	kms::to_clusters(labels, K, clusters);

	cout << "Silhouette score: " <<
		kms::silhouette_score(df, clusters, centers, K) << endl;

	return labels;
}

void KMeans::getCenters(RowVectorXd*& copy)
{
	for (int i = 0; i < K; i++)
		copy[i] = centers[i];
}

void KMeans::getClusters(vector<vector<int>>& copy) { copy = groups; }

void kms::single_fit(const MatrixXd& df, vector<vector<int>>& groups, RowVectorXd*& centers,
	int K, int init, int seed)
{
	if (init != 0 && init != 1)
	{
		cout << "Error(kms::single_fit(const MatrixXd&, RowVectorXd*&, int, int)): " <<
			"Invalid initialization." << endl;
		return;
	}
	else if (init == 0)
		rand_init_center(df, centers, K, seed);
	else
		kmpp_init_center(df, centers, K, seed);

	vector<vector<int>> temp(K, vector<int>());

	while (1)
	{
		assign_group(df, temp, centers, K);

		if (isSame(groups, temp))
			break;

		update_center(df, temp, centers);

		groups = temp;
		for (int i = 0; i < K; i++)
			temp[i].clear();
	}
}

void kms::rand_init_center(const MatrixXd& df, RowVectorXd*& centers, int K, int seed = 0)
{
	srand((unsigned)time(NULL) + seed);

	int* unique = new int[K];
	for (int i = 0; i < K; i++)
		centers[i] = df.row(unique_random(unique, K, (int)df.rows()));
}

int kms::unique_random(const int* unique, int size, int range)
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

void kms::kmpp_init_center(const MatrixXd& df, RowVectorXd*& centers, int K, int seed = 0)
{
	srand((unsigned)time(NULL) + seed);

	int idx = rand() % df.rows();
	centers[0] = df.row(idx);

	for (int i = 1; i < K; i++)		// because the first center is already assigned
	{
		vector<double> dists;
		for (int j = 0; j < df.rows(); j++)
		{
			idx = nearest_center(df.row(j), centers, i);
			dists.push_back(euclidean_norm(df.row(j), centers[idx]));
		}

		idx = std::distance(dists.begin(), std::max_element(dists.begin(), dists.end()));
		centers[i] = df.row(idx);
	}
}

int kms::nearest_center(const RowVectorXd& X, const RowVectorXd* centers, int K)
{
	int idx = K - 1;
	double min = euclidean_norm(X, centers[K - 1]);
	for (int i = 0; i < K - 1; i++)
	{
		double norm = euclidean_norm(X, centers[i]);
		if (min > norm)
		{
			min = norm;
			idx = i;
		}
	}
	return idx;
}

double kms::euclidean_norm(const RowVectorXd& p1, const RowVectorXd& p2)
{
	if (p1.size() != p2.size())
	{
		cout << "Error(KMeans::euclidean_norm(cosnt RowVectorXd&, const RowVectorXd&)): " <<
			"Vectors are not compatible." << endl;
		return 0.0;
	}

	double sum = 0;
	for (int i = 0; i < p1.size(); i++)
		sum += std::pow((p1[i] - p2[i]), 2);

	return std::sqrt(sum);
}

void kms::assign_group(const MatrixXd& df, vector<vector<int>>& groups,
	const RowVectorXd* centers, int K)
{
	for (int i = 0; i < df.rows(); i++)
	{
		int idx = nearest_center(df.row(i), centers, K);
		groups[idx].push_back(i);
	}
}

bool kms::isSame(const vector<vector<int>>& before, const vector<vector<int>>& current)
{
	if (before.size() == 0)
		return false;

	bool same = true;
	for (int i = 0; i < before.size(); i++)
	{
		if (before[i].size() == current[i].size())
		{
			for (int j = 0; j < before[i].size(); j++)
				if (before[i][j] != current[i][j])
				{
					same = false;
					break;
				}
			if (!same)
				break;
		}
		else
		{
			same = false;
			break;
		}
	}
	return same;
}

void kms::update_center(const MatrixXd& df, const vector<vector<int>>& groups,
	RowVectorXd*& centers)
{
	for (int i = 0; i < groups.size(); i++)
	{
		RowVectorXd sum;
		for (int j = 0; j < groups[i].size(); j++)
		{
			if (j == 0)
				sum = df.row(groups[i][j]);
			else
				sum += df.row(groups[i][j]);
		}
		centers[i] = sum / (double)groups[i].size();
	}
}

void kms::ensemble_fit(const MatrixXd& df, vector<vector<int>>& groups, RowVectorXd*& centers,
	int K, int init, int n_init)
{
	vector<RowVectorXd*> container(n_init);

	int min = 0;
	double bef, cur;
	for (int i = 0; i < n_init; i++)
	{
		RowVectorXd* temp_centers = new RowVectorXd[K];
		single_fit(df, groups, temp_centers, K, init, i + 1);
		cur = squared_error(df, groups, temp_centers);

		if (i == 0)
			bef = cur;
		else if (bef > cur)
		{
			min = i;
			bef = cur;
		}

		container[i] = temp_centers;
	}

	for (int i = 0; i < K; i++)
		centers[i] = container[min][i];

	for (RowVectorXd*& temp_centers : container)
		delete[] temp_centers;
}

double kms::squared_error(const MatrixXd& df, const vector<vector<int>>& groups,
	const RowVectorXd* centers)
{
	double error = 0;
	for (unsigned i = 0; i < groups.size(); i++)
		for (const int& idx : groups[i])
			error += std::pow(euclidean_norm(df.row(idx), centers[i]), 2);
	return error;
}

void kms::to_clusters(const VectorXd& labels, int K, vector<vector<int>>& clusters)
{
	clusters.resize(K, vector<int>());
	for (int i = 0; i < labels.rows(); i++)
		clusters[labels[i]].push_back(i);
}

double kms::silhouette_score(const MatrixXd& df, const vector<vector<int>>& groups,
	const RowVectorXd* centers, int K)
{
	double s = 0;
	for (int i = 0; i < groups.size(); i++)
		for (const int& point : groups[i])
		{
			double a = -1, b = -1;
			for (int j = 0; j < K; j++)
			{
				double temp = mean_distance(df, point, groups[j]);

				if (i == j)
					a = temp;
				else if ((b == -1) || (b > temp))
					b = temp;
			}
			s += (b - a) / std::max(a, b);
		}
	return s / df.rows();
}

double kms::mean_distance(const MatrixXd& df, int idx, const vector<int>& group)
{
	double sum = 0;
	for (int i = 0; i < group.size(); i++)
		sum += euclidean_norm(df.row(idx), df.row(group[i]));
	return sum / group.size();
}