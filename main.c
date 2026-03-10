#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// ==========================================
// 1. stories15M 模型的固定超参数
// ==========================================
#define DIM 288
#define HIDDEN_DIM 768
#define N_LAYERS 6
#define N_HEADS 6
#define N_KV_HEADS 6
#define VOCAB_SIZE 32000
#define SEQ_LEN 256
#define HEAD_SIZE 48 // DIM / N_HEADS (288 / 6)

// ==========================================
// 2. 权重与状态的原始数组定义 (全局分配以防爆栈)
// ==========================================
float token_embedding_table[VOCAB_SIZE][DIM];
float rms_att_weight[N_LAYERS][DIM];
float wq[N_LAYERS][DIM][DIM];
float wk[N_LAYERS][DIM][DIM];
float wv[N_LAYERS][DIM][DIM];
float wo[N_LAYERS][DIM][DIM];
float rms_ffn_weight[N_LAYERS][DIM];
float w1[N_LAYERS][HIDDEN_DIM][DIM];
float w2[N_LAYERS][DIM][HIDDEN_DIM];
float w3[N_LAYERS][HIDDEN_DIM][DIM];
float rms_final_weight[DIM];
float wcls[VOCAB_SIZE][DIM]; 

// KV Cache 缓存状态
float key_cache[N_LAYERS][SEQ_LEN][DIM];
float value_cache[N_LAYERS][SEQ_LEN][DIM];


// ==========================================
// 3. 最纯粹的基础算子 (无任何过度封装)
// ==========================================

// RMSNorm 正则化
void rmsnorm(float out[DIM], float x[DIM], float weight[DIM]) {
    float ss = 0.0f;
    for (int j = 0; j < DIM; j++) {
        ss += x[j] * x[j];
    }
    ss /= DIM;
    ss += 1e-5f; // epsilon
    ss = 1.0f / sqrtf(ss);
    
    for (int j = 0; j < DIM; j++) {
        out[j] = weight[j] * (ss * x[j]);
    }
}

// 矩阵乘法: y = x * W  (W的形状为 [输出维度][输入维度])
void matmul_dim_dim(float out[DIM], float x[DIM], float w[DIM][DIM]) {
    for (int i = 0; i < DIM; i++) {
        float val = 0.0f;
        for (int j = 0; j < DIM; j++) {
            val += w[i][j] * x[j];
        }
        out[i] = val;
    }
}

void matmul_hidden_dim(float out[HIDDEN_DIM], float x[DIM], float w[HIDDEN_DIM][DIM]) {
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float val = 0.0f;
        for (int j = 0; j < DIM; j++) {
            val += w[i][j] * x[j];
        }
        out[i] = val;
    }
}

void matmul_dim_hidden(float out[DIM], float x[HIDDEN_DIM], float w[DIM][HIDDEN_DIM]) {
    for (int i = 0; i < DIM; i++) {
        float val = 0.0f;
        for (int j = 0; j < HIDDEN_DIM; j++) {
            val += w[i][j] * x[j];
        }
        out[i] = val;
    }
}

// RoPE 旋转位置编码 (实时计算三角函数，剔除 precomputed 预计算表)
void apply_rope(float q[DIM], float k[DIM], int pos) {
    for (int i = 0; i < DIM; i += 2) {
        int head_dim = i % HEAD_SIZE;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)HEAD_SIZE);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        float q0 = q[i], q1 = q[i+1];
        q[i]   = q0 * fcr - q1 * fci;
        q[i+1] = q0 * fci + q1 * fcr;

        float k0 = k[i], k1 = k[i+1];
        k[i]   = k0 * fcr - k1 * fci;
        k[i+1] = k0 * fci + k1 * fcr;
    }
}

// 多头自注意力机制 (Causal Multi-Head Attention)
void attention(float out[DIM], int layer, int pos, float q[DIM], 
               float k_cache[N_LAYERS][SEQ_LEN][DIM], 
               float v_cache[N_LAYERS][SEQ_LEN][DIM]) {
                   
    float att[N_HEADS][SEQ_LEN] = {0}; // 注意力分数矩阵

    for (int h = 0; h < N_HEADS; h++) {
        int head_offset = h * HEAD_SIZE;

        // 1. 计算 Q * K^T
        for (int t = 0; t <= pos; t++) {
            float score = 0.0f;
            for (int i = 0; i < HEAD_SIZE; i++) {
                score += q[head_offset + i] * k_cache[layer][t][head_offset + i];
            }
            score /= sqrtf((float)HEAD_SIZE);
            att[h][t] = score;
        }

        // 2. Softmax 操作
        float max_val = att[h][0];
        for (int t = 1; t <= pos; t++) {
            if (att[h][t] > max_val) max_val = att[h][t];
        }
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) {
            att[h][t] = expf(att[h][t] - max_val);
            sum += att[h][t];
        }
        for (int t = 0; t <= pos; t++) {
            att[h][t] /= sum;
        }

        // 3. 计算 Softmax * V
        for (int i = 0; i < HEAD_SIZE; i++) {
            float val = 0.0f;
            for (int t = 0; t <= pos; t++) {
                val += att[h][t] * v_cache[layer][t][head_offset + i];
            }
            out[head_offset + i] = val;
        }
    }
}

// SwiGLU 前馈神经网络
void ffn(float out[DIM], float x[DIM], float l_w1[HIDDEN_DIM][DIM], 
         float l_w2[DIM][HIDDEN_DIM], float l_w3[HIDDEN_DIM][DIM]) {
             
    float h1[HIDDEN_DIM];
    float h2[HIDDEN_DIM];

    matmul_hidden_dim(h1, x, l_w1); // h1 = x * w1
    matmul_hidden_dim(h2, x, l_w3); // h2 = x * w3

    // SwiGLU 激活: h1 = h1 * SiLU(h1) * h2
    for (int i = 0; i < HIDDEN_DIM; i++) {
        float val = h1[i];
        val *= (1.0f / (1.0f + expf(-val))); // SiLU = x * sigmoid(x)
        val *= h2[i];
        h1[i] = val;
    }

    matmul_dim_hidden(out, h1, l_w2); // out = h1 * w2
}

// ==========================================
// 4. 核心前向传播 (将所有多维数组一目了然地传入)
// ==========================================
void forward(
    int token, int pos,
    float p_token_embedding_table[VOCAB_SIZE][DIM],
    float p_rms_att_weight[N_LAYERS][DIM],
    float p_wq[N_LAYERS][DIM][DIM],
    float p_wk[N_LAYERS][DIM][DIM],
    float p_wv[N_LAYERS][DIM][DIM],
    float p_wo[N_LAYERS][DIM][DIM],
    float p_rms_ffn_weight[N_LAYERS][DIM],
    float p_w1[N_LAYERS][HIDDEN_DIM][DIM],
    float p_w2[N_LAYERS][DIM][HIDDEN_DIM],
    float p_w3[N_LAYERS][HIDDEN_DIM][DIM],
    float p_rms_final_weight[DIM],
    float p_wcls[VOCAB_SIZE][DIM],
    float p_key_cache[N_LAYERS][SEQ_LEN][DIM],
    float p_value_cache[N_LAYERS][SEQ_LEN][DIM],
    float logits[VOCAB_SIZE],
    float x[DIM] // 当前输入状态
) {
    // 1. 获取 Token 的词嵌入
    for (int i = 0; i < DIM; i++) {
        x[i] = p_token_embedding_table[token][i];
    }

    float xb[DIM], xb2[DIM];
    float q[DIM], k[DIM], v[DIM];

    // 2. 遍历所有 Transformer 层
    for (int l = 0; l < N_LAYERS; l++) {
        // --- 注意力模块 ---
        rmsnorm(xb, x, p_rms_att_weight[l]);
        
        matmul_dim_dim(q, xb, p_wq[l]);
        matmul_dim_dim(k, xb, p_wk[l]);
        matmul_dim_dim(v, xb, p_wv[l]);
        
        apply_rope(q, k, pos);

        // 写入 KV 缓存
        for (int i = 0; i < DIM; i++) {
            p_key_cache[l][pos][i] = k[i];
            p_value_cache[l][pos][i] = v[i];
        }

        attention(xb2, l, pos, q, p_key_cache, p_value_cache);
        matmul_dim_dim(xb, xb2, p_wo[l]);

        // 残差连接
        for (int i = 0; i < DIM; i++) x[i] += xb[i];

        // --- 前馈网络模块 (FFN) ---
        rmsnorm(xb, x, p_rms_ffn_weight[l]);
        ffn(xb2, xb, p_w1[l], p_w2[l], p_w3[l]);

        // 残差连接
        for (int i = 0; i < DIM; i++) x[i] += xb2[i];
    }

    // 3. 最后的输出映射
    rmsnorm(xb, x, p_rms_final_weight);

    for (int i = 0; i < VOCAB_SIZE; i++) {
        float val = 0.0f;
        for (int j = 0; j < DIM; j++) {
            val += p_wcls[i][j] * xb[j];
        }
        logits[i] = val;
    }
}

// ==========================================
// 5. 暴力读取 bin 文件
// ==========================================
void load_weights(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("找不到模型文件!\n"); exit(1); }

    // 跳过 Karpathy 设置的 28 bytes 头部信息 (7 个 int)
    fseek(f, 28, SEEK_SET);

    // 严格按顺序纯二进制暴力灌入多维数组
    fread(token_embedding_table, sizeof(float), VOCAB_SIZE * DIM, f);
    fread(rms_att_weight, sizeof(float), N_LAYERS * DIM, f);
    fread(wq, sizeof(float), N_LAYERS * DIM * DIM, f);
    fread(wk, sizeof(float), N_LAYERS * DIM * DIM, f);
    fread(wv, sizeof(float), N_LAYERS * DIM * DIM, f);
    fread(wo, sizeof(float), N_LAYERS * DIM * DIM, f);
    fread(rms_ffn_weight, sizeof(float), N_LAYERS * DIM, f);
    fread(w1, sizeof(float), N_LAYERS * HIDDEN_DIM * DIM, f);
    fread(w2, sizeof(float), N_LAYERS * DIM * HIDDEN_DIM, f);
    fread(w3, sizeof(float), N_LAYERS * HIDDEN_DIM * DIM, f);
    fread(rms_final_weight, sizeof(float), DIM, f);
/*
    // 检查是否有独立的 wcls，如果读不到说明与 embedding 共享权重
    if (fread(wcls, sizeof(float), VOCAB_SIZE * DIM, f) == 0) {
        for (int i = 0; i < VOCAB_SIZE; i++)
            for (int j = 0; j < DIM; j++)
                wcls[i][j] = token_embedding_table[i][j];
    }

    fclose(f);*/

    // 强制复用 Embedding 权重 (stories15M 标准行为)
    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int j = 0; j < DIM; j++) {
            wcls[i][j] = token_embedding_table[i][j];
        }
    }

    fclose(f);
}

void check_header(const char* filename) {
    FILE* f = fopen(filename, "rb");
    int config[7];
    fread(config, sizeof(int), 7, f);
    fclose(f);
    printf("--- 模型 Header 检查 ---\n");
    printf("DIM: %d\n", config[0]);
    printf("HIDDEN_DIM: %d\n", config[1]);
    printf("N_LAYERS: %d\n", config[2]);
    printf("N_HEADS: %d\n", config[3]);
    printf("N_KV_HEADS: %d\n", config[4]);
    printf("VOCAB_SIZE: %d\n", config[5]);
    printf("SEQ_LEN: %d\n", config[6]);
}
// 简单的内存加载 tokenizer (无任何封装)
char* vocab[VOCAB_SIZE];
void load_tokenizer(const char* filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return;
    int max_token_length;
    fread(&max_token_length, sizeof(int), 1, f);
    for (int i = 0; i < VOCAB_SIZE; i++) {
        float score;
        fread(&score, sizeof(float), 1, f);
        int len;
        fread(&len, sizeof(int), 1, f);
        vocab[i] = (char*)malloc(len + 1);
        fread(vocab[i], 1, len, f);
        vocab[i][len] = '\0';
    }
    fclose(f);
}

// ==========================================
// 6. 随机采样与温度控制 (拯救乱码的核心)
// ==========================================
int sample_with_temperature(float logits[VOCAB_SIZE], float temperature) {
    // 1. 找最大值 (防止指数爆炸溢出)
    float max_val = logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    // 2. 引入温度计算指数并求和
    float sum = 0.0f;
    float probs[VOCAB_SIZE];
    for (int i = 0; i < VOCAB_SIZE; i++) {
        // Temperature 越大，概率分布越平缓；越小，越接近绝对 Argmax
        probs[i] = expf((logits[i] - max_val) / temperature);
        sum += probs[i];
    }

    // 3. 轮盘赌采样 (Roulette Wheel Selection)
    float r = ((float)rand() / (float)RAND_MAX) * sum;
    float cdf = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        cdf += probs[i];
        if (r <= cdf) {
            return i;
        }
    }
    return VOCAB_SIZE - 1; // 理论上走不到这里，兜底
}

int main() {
    srand((unsigned int)time(NULL));

    printf("正在加载 60MB 权重...\n");
    load_weights("stories15M.bin");
    
    printf("正在加载 Tokenizer...\n");
    load_tokenizer("tokenizer.bin");
    
    float current_x[DIM] = {0};
    float logits[VOCAB_SIZE] = {0};
    int current_token = 1; // 1 = <s> (序列开始)
    
    printf("\n>>> 纯 C 语言无依赖大模型开始生成 <<<\n\n");
    
    // 自回归生成 50 个 Token
    for (int pos = 0; pos < 50; pos++) {
        // 1. 前向传播
        forward(current_token, pos,
                token_embedding_table, rms_att_weight, wq, wk, wv, wo,
                rms_ffn_weight, w1, w2, w3, rms_final_weight, wcls,
                key_cache, value_cache, logits, current_x);

        // 2. 贪婪解码 (找出得分最高的 Token)
        float max_val = logits[0];
        // 2. 带温度的采样 (0.9 左右是个讲故事的好温度)
        int next_token = sample_with_temperature(logits, 0.9f); 
        // 3. 打印人类可读的词汇
        if (vocab[next_token]) {
            // Llama2 的词表中，空格被编码为 ' ' (即 e2 96 81)
            printf("%s", vocab[next_token]); 
            fflush(stdout); // 强制刷新缓冲区，实现打字机效果
        }

        // 4. 将输出作为下一步的输入 (自回归的核心!)
        current_token = next_token;
    }
    
    printf("\n\n>>> 生成结束 <<<\n");
    return 0;
}
