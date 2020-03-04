#include <iostream>
#include <Eigen/dense>
#include "GMM.h"
#include "file_manage.h"
using namespace std;
using namespace Eigen;

int main()
{
	MatrixXd df;
	VectorXd labels;

	read_csv("data/iris.csv", df, labels, 150, 5);
	//read_csv("data/seeds.csv", df, labels, 210, 8);

	GMM model(3);
	model.fit(df, 1);
	model.predict(df);

	return 0;
}