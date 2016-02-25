#include <QCoreApplication>

#include "helper/helper.h"

void test1();
void test2();

#include <iostream>

int main(int argc, char *argv[])
{
	hlp::unused(argc, argv);
	std::cout << "Running test1...";
	test1();
	std::cout << "done." << std::endl;
	std::cout << "Running test2...";
	test2();
	std::cout << "done." << std::endl;
}
