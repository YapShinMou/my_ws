#include <iostream>
#include "user_def_lib.hpp"

int CCC = 0;

class classA {
public:
	int AA;
	int BB;
	
	classA (int A, int B) //建構子(constructor)
	{
		AA = A;
		BB = B;
	}
	
	void func() {
//		CCC = 1;
		std::cout << CCC << std::endl;
	}
	
private:
	int CCC = 3;
};

int main()
{
	classA classtest(1, 2);
	
	std::cout << classtest.AA << ", " << classtest.BB << std::endl;
	std::cout << add(1, 2) << std::endl;
	
	classtest.func();
	std::cout << CCC << std::endl;
	
	return 0;
}

