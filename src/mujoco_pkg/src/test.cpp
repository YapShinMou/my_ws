#include <iostream>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

// 檢查是否有輸入（非阻塞）
int kbhit(void) {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);             // 取得目前終端設定
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);           // 關閉行緩衝與回顯
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);  // 設為非阻塞模式

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);    // 恢復終端設定
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if(ch != EOF) {
        ungetc(ch, stdin);                      // 將字元放回輸入緩衝
        return 1;
    }

    return 0;
}

// 直接取得一個字元，不需按 Enter
char getch() {
    char buf = 0;
    struct termios old = {0};
    if (tcgetattr(STDIN_FILENO, &old) < 0)
        perror("tcsetattr()");
    old.c_lflag &= ~(ICANON | ECHO); // 關閉行緩衝與回顯
    if (tcsetattr(STDIN_FILENO, TCSANOW, &old) < 0)
        perror("tcsetattr ICANON");
    if (read(STDIN_FILENO, &buf, 1) < 0)
        perror("read()");
    old.c_lflag |= (ICANON | ECHO);
    if (tcsetattr(STDIN_FILENO, TCSADRAIN, &old) < 0)
        perror("tcsetattr ~ICANON");
    return buf;
}

int main() {
    std::cout << "按 q 離開 (不用按 Enter)\n";

    while (true) {
        if (kbhit()) { // 有鍵盤輸入才執行
            char c = getch();
            std::cout << "你按了: " << c << std::endl;
            if (c == 'q') break;
        }
        usleep(10000); // 稍作等待 (10ms)
    }

    std::cout << "結束\n";
    return 0;
}

