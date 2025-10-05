# 範例
Mujoco模擬範例程式、MJCF範例、建構子範例、自訂函式庫、CMakeLists範例、README範例

## Build
```bash
cd ~/my_ws
mkdir build && cd build
cmake ..
```
編譯
```bash
cd ~/my_ws/build
make
```

## Run
```bash
cd ~/my_ws/build/src/mujoco_pkg
./CartPole
```
```bash
cd ~/my_ws/build/src/mujoco_pkg
./CartPole_without_GUI
```

## 結構
```bash
my_ws/
├── CMakeLists.txt
└── src/
	├── mujoco_pkg/
	|	├── CMakeLists.txt
	|	├── models/
	|	│	├── cart_pole.xml
	|	│	├── cube.xml
	|	|	└── dog.xml
	|	└── src/
	|		├── CartPole.cpp
	|		└── CartPole_without_GUI.cpp
	└── practise_pkg/
		├── CMakeLists.txt
		├── README.md
		├── include/
		|	└── user_def_lib.hpp
		└── src/
			├── constructor_lib_example.cpp
			└── user_def_lib.cpp
```
