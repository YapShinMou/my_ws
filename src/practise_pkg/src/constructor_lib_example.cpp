#include <iostream>
#include "user_def_lib.hpp"

class classA {
public:
	int AA;
	int BB;
	
	classA (int A, int B) //建構子(constructor)
	{
		AA = A;
		BB = B;
	}
};

int main()
{
	classA classtest(1, 2);
	
	std::cout << classtest.AA << ", " << classtest.BB << std::endl;
	std::cout << add(1, 2) << std::endl;
	
	return 0;
}

