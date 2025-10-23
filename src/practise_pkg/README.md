**自製函式庫、建構子(constructor)範例**

函式內宣告的變數只能在該函式使用

函式內宣告的物件只能在該函式使用

for、if、while內宣告的變數只能在該迴圈使用

for、if、while內宣告的物件只能在該迴圈使用

全域(函式外)不能呼叫函式，不能執行for、if、while

class裡面的任一函式可以呼叫任一class裡面宣告的變數和函式

# 專案名稱 (Project Title) README範例
簡短描述你的專案是做什麼的（1～3 行）。

## 目錄 (Table of Contents)
- [簡介 (Introduction)](#簡介-introduction)
- [安裝 (Installation)](#安裝-installation)
- [使用方式 (Usage)](#使用方式-usage)
- [目錄結構 (Directory Structure)](#目錄結構-directory-structure)
- [注意事項 (Notes)](#注意事項-notes)
- [授權 (License)](#授權-license)

## 簡介 (Introduction)
說明背景、用途或動機。
### A
- 中
- 中
- 中
### B
**粗體字**

*斜體字*

## 安裝 (Installation)
```bash
git clone https://github.com/yourname/repository.git
cd repository
```

## 使用方式 (Usage)

## 目錄結構 (Directory Structure)
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
| 名稱 | 說明 |
|------|------|
| A    | 內容 |
| B    | 內容 |

## 注意事項 (Notes)

## 授權 (License)
