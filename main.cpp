#include <iostream>
#include <Eigen/dense>
#include "file_manage.h"
#include "gaussian_mixture.h"
using namespace std;
using namespace Eigen;

int main()
{
	MatrixXd features;
	VectorXd labels;

	read_csv("data/iris.csv", features, labels, 150, 5);
	//read_csv("data/seeds.csv", features, labels, 210, 8);

	GMM model(3);
	model.fit(features, 1);

	return 0;
}