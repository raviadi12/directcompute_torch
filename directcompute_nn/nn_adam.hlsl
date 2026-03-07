cbuffer Params : register(b0) { 
    uint count; 
    float step_size;
    float beta1; 
    float beta2; 
    float eps; 
    float weight_decay; 
    float clip; 
    uint pad; 
};

RWStructuredBuffer<float> weight : register(u0);
RWStructuredBuffer<float> m : register(u1);
RWStructuredBuffer<float> v : register(u2);
StructuredBuffer<float> grad : register(t0);

[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < count) {
        float g = grad[id.x];
        if (g > clip) g = clip;
        if (g < -clip) g = -clip;
        
        float current_w = weight[id.x];
        float m_t = beta1 * m[id.x] + (1.0 - beta1) * g;
        float v_t = beta2 * v[id.x] + (1.0 - beta2) * g * g;
        
        m[id.x] = m_t;
        v[id.x] = v_t;
        
        // step_size already includes the bias correction from python
        // weight_decay already has lr factor applied if necessary (AdamW: lr*wd, Adam: 0 inside here)
        weight[id.x] = current_w - weight_decay * current_w - step_size * m_t / (sqrt(v_t) + eps);
    }
}
