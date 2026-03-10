#include <stdio.h>
#include <stdlib.h>

#define VOCAB_SIZE 32000

int main() {
    // 请确保目录下有从 Karpathy 仓库下载的 tokenizer.bin
    FILE *f = fopen("tokenizer.bin", "rb");
    if (!f) { 
        printf("找不到 tokenizer.bin 文件!\n"); 
        return 1; 
    }

    int max_token_length;
    fread(&max_token_length, sizeof(int), 1, f);

    for (int i = 0; i < VOCAB_SIZE; i++) {
        float score;
        fread(&score, sizeof(float), 1, f); // 读取词频得分
        
        int len;
        fread(&len, sizeof(int), 1, f); // 读取字符串长度
        
        char word[256] = {0}; // 假设单个 Token 长度不超过 256
        fread(word, 1, len, f); // 读取真正的字符
        
        if (i == 26) {
            printf("\n>>> 揭晓时刻: Token ID 26 的解码结果是: [%s] <<<\n\n", word);
            break; // 找到了就退出
        }
    }
    
    fclose(f);
    return 0;
}
