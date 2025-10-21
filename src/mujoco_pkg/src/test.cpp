#include <thread>
#include <mutex>
#include <vector>
#include <deque>
#include <random>
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

// ==================== Replay Buffer ========================
class ReplayBuffer {
public:
	//自訂資料型態
	struct Experience {
		int state;
		int action;
		int reward;
		int next_state;
	};
	
	void push(int s, int a, int r, int s2) {
		if (buffer_.size() >= 30) buffer_.pop_front();
		buffer_.push_back({s, a, r, s2});
	}
	
	std::vector<Experience> sample() {
		std::vector<Experience> batch; //宣告一個名稱叫做 batch 的變數，型態為 std::vector<Experience>
		std::uniform_int_distribution<int> dist(0, buffer_.size() - 1); //dist(rng) 為0到buffer_.size() - 1隨機整數
		for (int i=0; i<10; i++) {
			batch.push_back(buffer_[dist(rng)]);
		}
		return batch;
	}
	
	int size() const { return buffer_.size(); }
	
private:
	std::deque<Experience> buffer_; //宣告一個名稱叫做 buffer_ 的變數，型態為 std::deque<Experience>，頭尾兩端插入及刪除十分快速的陣列，元素型態為 Experience
	std::mt19937 rng{std::random_device{}()}; //建立一個亂數引擎
};

int main()
{
	ReplayBuffer memory;
	for (int i = 0 ; i < 21 ; ++i) {
		memory.push(i, i, i, i);
	}
	auto batch = memory.sample();
	for (const auto& exp : batch) {
    std::cout << "state=" << exp.state
              << ", action=" << exp.action
              << ", reward=" << exp.reward
              << ", next_state=" << exp.next_state
              << std::endl;
}
std::cout << memory.size() << std::endl;

}	











